import os
import shutil
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from tqdm import tqdm
import wandb

# Import from sibling directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + ".")
from .metrics import Metrics
from .survival_loss import NLLDeepSurvLoss, predict_survival
from .surv_cnn import create_survival_resnet50, create_survival_sparsebagnet33, create_survival_inceptionV3
from .breslow_estimator import fit_breslow, init_breslow
from .lr_scheduler import create_scheduler

# Import from parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "..")
from utils.cnn_utils import (
    get_lr,
    get_model_filename,
    get_cuda,
    optimizer_to_device,
    get_short_model_name,
)
from data.cnn_surv_dataloader import (
    get_train_loader_surv,
    get_val_loader_surv,
    get_test_loader_surv,
    get_dataset
)
from utils.cnn_survival_utils import e_t_to_tuple, get_event_indicator_matrix
from utils.cnn_wandb import get_run_id
from utils.helpers import get_project_dir, parse_list, set_seed

EPSILON = 1e-7

def get_loader(c: SimpleNamespace, split: str):
    surv = True if "surv" in c.cnn.network.lower() else False
    if not surv:
        raise ValueError("Only survival training is supported!")
    suffix = "_surv" if surv else ""
    return eval(f"get_{split}_loader{suffix}(c)")

class CNN:
    """CNN class"""

    def __init__(self, c: SimpleNamespace, estimator: Metrics):
        self.c = c

        print(f"Model will run on GPU {self.c.cnn.gpu}\n")

        if hasattr(self.c.cnn, "use_stereo_pairs") and self.c.cnn.use_stereo_pairs:
            self.use_stereo_pairs = True
            print("Using stereo pairs.")
        else:
            self.use_stereo_pairs = False

        set_seed(self.c.cnn.seed)

        self.model = self._create_model()
        self.train_loader = get_loader(self.c, "train")
        self.optimizer = self._create_optimizer()
        self.scheduler, self.warmup_scheduler = create_scheduler(self)

        self.estimator = estimator

        self._init_vars()

        # Use passed run id or generate a new one
        self.c.cnn.run_id = get_run_id(self.c)

        self.init_wandb()

        self.filename = get_model_filename(self.c)

        self.checkpoint_path = os.path.join(
            get_project_dir(), "checkpoints", self.filename
        )
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        self.breslow = None

        if hasattr(self.c.cnn, "sparsity_lambda"):
            if isinstance(self.c.cnn.sparsity_lambda, (float, int)) and self.c.cnn.sparsity_lambda > 0:
                self.sparsity_lambda = self.c.cnn.sparsity_lambda
        
        print(f"Run id: {self.c.cnn.run_id}\n")

        if hasattr(self.c.cnn, "load_best_model") and self.c.cnn.load_best_model:
            self.load_checkpoint(best=True)
        elif hasattr(self.c.cnn, "resume_training") and self.c.cnn.resume_training:
            self.load_checkpoint(best=False)

        self.y_train, self.y_val = self._get_datasets()


    def init_wandb(self):
        # Check if wandb is initialized already (e.g. when running a sweep)
        if wandb.run is not None:
            print(f"Found wandb.run. Config: {vars(self.c.cnn)}\n\n")
            wandb.run.name = get_short_model_name(self.c)
            if hasattr(self.c.cnn, "tags") and self.c.cnn.tags is not None:
                wandb.run.tags = parse_list(self.c.cnn.tags)
            if hasattr(wandb.run, "id"):# and wandb.run.id is not None:
                self.c.cnn.run_id = wandb.run.id

            wandb.config.update(dict(vars(self.c.cnn)))
        else:
            # Init wandb run
            print(f"Initializing wandb run. Config: {vars(self.c.cnn)}\n\n")
            wandb.init(
                project=self.c.cnn.project,
                name=get_short_model_name(self.c),
                config=vars(self.c.cnn),
                tags=parse_list(self.c.cnn.tags)
                if "tags" in vars(self.c.cnn) and self.c.cnn.tags is not None
                else None,
                resume="allow",
                id=self.c.cnn.run_id,
            )

    def _init_vars(self):
        """Initialize variables"""

        def _init_model_selection_score():
            if (
                "loss" in self.c.cnn.model_selection_metric.lower()
                or "ibs" in self.c.cnn.model_selection_metric.lower()
            ):
                self.best_score = np.inf
            else:
                self.best_score = -1.0

        def _init_performance_dict():
            splits = ["val", "test"]
            self.performances = {split: {} for split in splits}
            for split in splits:
                self.performances[split] = {self.c.cnn.num_classes: {}, 2: {}}

        self.current_batch = 0
        self.current_epoch = 0
        self.num_bad_epochs = 0
        self.best_epoch = 0
        self.best_score = None
        self.is_best = False
        _init_model_selection_score()
        _init_performance_dict()

        if hasattr(self.c.cnn, "resume_training") and self.c.cnn.resume_training or hasattr(self.c.cnn, "load_best_model") and self.c.cnn.load_best_model:
            resume = True
        else:
            resume = False

        self.c.cnn.run_id = (
            None if not resume else self.c.cnn.run_id
        )
        
        self.device = get_cuda(self.c.cnn.gpu)
        self.model.to(self.device)
    
    def _get_datasets(self): # Dependency on training data. Could load this from a file instead in the future.
        y_train_set = get_dataset(split="train", c=self.c).get_e_t()
        y_val_set = get_dataset(split="val", c=self.c).get_e_t()

        y_train = e_t_to_tuple(
            y_train_set[0],
            y_train_set[1],
        )
        y_val = e_t_to_tuple(
            y_val_set[0],
            y_val_set[1],
        )
        return y_train, y_val

    def _create_model(self):
        """Create model"""

        if any([loss in self.c.cnn.loss.lower() for loss in ["cox", "deepsurv"]]):
            # Use model with only one output (hazard ratio at a time)
            self.type = "cox"
            self.c.cnn.num_outputs = 1
        elif any([loss in self.c.cnn.loss.lower() for loss in ["nnet", "logistic"]]):
            # Use model that outputs one hazard for each time in survival_times
            self.type = "nnet"
            self.c.cnn.num_outputs = len(self.c.cnn.survival_times)
        elif any([loss in self.c.cnn.loss.lower() for loss in ["clf", "celoss", "bce", "classification", "crossentropy"]]):
            # Use model that outputs one hazard for the one inquired time
            self.type = "classification"
            self.c.cnn.num_outputs = 1
            assert len(self.c.cnn.survival_times) == 1, "Classification only supports one survival time!"
        else:
            raise ValueError(f"Invalid loss type: {self.c.cnn.loss}!")
        
        print(f"Using {self.type} model.")
        
        print(
            f"Creating model of type {self.c.cnn.network} with {self.c.cnn.num_outputs} outputs..."
        )

        if all([t in self.c.cnn.network.lower() for t in ["surv", "resnet"]]):
            model = create_survival_resnet50(self.c.cnn.num_outputs)
        
        elif all([t in self.c.cnn.network.lower() for t in ["surv", "inception"]]):
            model = create_survival_inceptionV3(self.c.cnn.num_outputs)

        elif all([t in self.c.cnn.network.lower() for t in ["surv", "bagnet"]]):
            model = create_survival_sparsebagnet33(self.c.cnn.num_outputs)

        else:
            raise ValueError(f"Invalid network type: {self.c.cnn.network}!")
        
        def get_survival_loss(out_hazards, e, t, activations=None):
            """Get survival loss of batch. out_hazards is a tensor of shape (batch_size, n_preds)
            or (batch_size, 2, n_preds) if using stereo pairs. e and t are tensors of shape (batch_size, )"""

            if self.type == "cox":
                nll_loss = NLLDeepSurvLoss().to(self.device)

                if out_hazards.shape[1] == 2:
                    # If using stereo pairs, average their losses
                    surv_loss1 = nll_loss(out_hazards[:, 0, :], t, e)
                    surv_loss2 = nll_loss(out_hazards[:, 1, :], t, e)
                    surv_loss = (surv_loss1 + surv_loss2) / 2
                else:
                    surv_loss = nll_loss(
                        out_hazards, t, e
                        )
                    
            elif self.type == "classification":
                event_indicator_matrix = get_event_indicator_matrix(events = e, durations = t).to(self.device)
                nll_loss = torch.nn.functional.binary_cross_entropy

                if out_hazards.shape[1] == 2:
                    # If using stereo pairs, average their survival predictions
                    risks1 = out_hazards[:, 0, :].sigmoid()
                    risks2 = out_hazards[:, 1, :].sigmoid()
                    risks = (risks1 + risks2) / 2
                else:
                    risks = out_hazards.sigmoid()

                labels = event_indicator_matrix[:,self.c.cnn.survival_times]
                surv_loss = nll_loss(risks, labels)
                
            else:
                raise ValueError(f"Invalid loss type: {self.c.cnn.loss}!")

            if hasattr(self, "sparsity_lambda"):
                loss = model.get_sparsity_loss(activations, lambda_=self.sparsity_lambda)
                surv_loss += loss

            return surv_loss

        self.get_survival_loss = get_survival_loss

        return model      

    def _create_optimizer(self):
        opt = self.c.cnn.optimizer.lower() if hasattr(self.c.cnn, "optimizer") else "adam"

        if opt == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.c.cnn.lr,
                momentum=0,
                weight_decay=self.c.cnn.weight_decay,
            )
        elif opt == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.c.cnn.lr,
                weight_decay=self.c.cnn.weight_decay,
            )
        elif opt == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.c.cnn.lr, weight_decay=self.c.cnn.weight_decay)
        else:
            raise ValueError(f"Invalid optimizer: {opt}!")
        
        return optimizer

    def _score_is_better(self, score):
        if (
            "loss" in self.c.cnn.model_selection_metric.lower()
            or "ibs" in self.c.cnn.model_selection_metric.lower()
        ):
            return score < self.best_score
        else:
            return score > self.best_score
        
    def get_survival_predictions(self, preds):
        """
        Get survival probabilities from model predictions
        
        Args:
            preds: model predictions (n, ) of all images in your dataset
        
        Returns:
            survs: survival probabilities (n, len(survival_times))
        """

        def _get_survival_preds_single(preds):
            if self.type == "nnet":
                hazards = preds.sigmoid().cpu()
                survs = (1.0 - hazards).add(EPSILON).log().cumsum(1).exp()

            elif self.type == "cox":
                preds = preds.cpu()
                times_to_predict = self.c.cnn.survival_times
                assert hasattr(self, "breslow"), "Breslow estimator must first be initialized or fitted!"
                survs = predict_survival(self.breslow, preds, times_to_predict)
                survs = torch.tensor(survs)
                if torch.isnan(survs).any():
                    print("WARNING: survs contains nan!")

            elif self.type == "classification":
                # Risk probs at the one inquired time
                risks = preds.sigmoid().cpu()
                survs = 1.0 - risks

            else:
                raise ValueError("Invalid loss/model type")

            return survs
        
        if preds.shape[1] == 2: # (bs, 2, n_preds)
            print("Stereo setup in get_survival_predictions")
            # If using stereo pairs, average the predictions post-activation
            preds1 = preds[:,0,:]
            preds2 = preds[:,1,:]
            survs1 = _get_survival_preds_single(preds1)
            survs2 = _get_survival_preds_single(preds2)
            survs = torch.stack([survs1, survs2]).mean(dim=0)

        else: # (bs, n_preds)
            survs = _get_survival_preds_single(preds)

        return survs

    def load_checkpoint(self, best: False):
        file_name = self.checkpoint_path

        file_base_name, file_extension = os.path.splitext(file_name)

        if best:
            file_name = file_base_name + "_best" + file_extension

        checkpoint = torch.load(file_name)

        self.current_batch = checkpoint["current_batch"] + 1
        self.current_epoch = checkpoint["current_epoch"] + 1

        if "best_score" in checkpoint:
            self.best_score = checkpoint["best_score"]

        if "num_bad_epochs" in checkpoint:
            self.num_bad_epochs = checkpoint["num_bad_epochs"]

        if "run_id" in checkpoint:
            self.c.cnn.run_id = checkpoint["run_id"]

        self.model.load_state_dict(checkpoint["model"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None:

            self.scheduler.load_state_dict(checkpoint["scheduler"])

            if self.scheduler.__class__.__name__ == "OneCycleLR":
                self.scheduler.last_epoch = self.current_batch

                scheduler_state = self.scheduler.state_dict()
                scheduler_state["total_steps"] = scheduler_state[
                    "total_steps"
                ] + self.c.cnn.num_epochs * int(len(self.train_loader))
                self.scheduler.load_state_dict(scheduler_state)
            elif self.scheduler.__class__.__name__ == "CosineAnnealingLR":
                self.scheduler.last_epoch = self.current_epoch
        
        # Discard warmup scheduler if loaded from checkpoint, i.e. don't attempt to resume a
        # training if warmup was not yet finished
        self.warmup_scheduler = None

        self.model.to(self.device)
        optimizer_to_device(self.optimizer, device=self.device)

        self.c.cnn.run_id = get_run_id(self.c)

        # Initialize Breslow Estimator
        if "breslow" in checkpoint and checkpoint["breslow"] is not None:
            self.breslow = init_breslow(
                checkpoint["breslow"]["cum_baseline_hazard"],
                checkpoint["breslow"]["baseline_survival"],
                checkpoint["breslow"]["unique_times"],
            )

        print(f"Loaded checkpoint: {file_name}")
        print(f"With run id: {self.c.cnn.run_id}")

    def save_checkpoint(self):
        file_name = self.checkpoint_path

        if self.breslow is not None:
            breslow = {
                "cum_baseline_hazard": self.breslow.cum_baseline_hazard_,
                "baseline_survival": self.breslow.baseline_survival_,
                "unique_times": self.breslow.unique_times_
                }
        else:
            breslow = None

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "current_epoch": self.current_epoch,
            "current_batch": self.current_batch,
            "is_best": self.is_best,
            "best_performance": self.best_score,
            "num_bad_epochs": self.num_bad_epochs,
            "run_id": self.c.cnn.run_id,
            "breslow": breslow
        }

        # Save
        file_base_name, file_extension = os.path.splitext(file_name)

        # Rename previous local checkpoint if already exists
        if os.path.exists(file_name):
            old_file_name = file_base_name + "_old" + file_extension
            os.rename(file_name, old_file_name)

        torch.save(checkpoint, file_name)

        if self.is_best:
            best_file_name = file_base_name + "_best" + file_extension
            shutil.copyfile(file_name, best_file_name)

        print(f"Saved checkpoint: {best_file_name if self.is_best else file_name}")

    def evaluate(
        self,
        split: str,
        best: bool = True,
        log_all: bool = True,
    ):
        """Evaluate model on a given split

        Args:
            split (str): Split to evaluate on
            best (bool, optional): Evaluate best model. If False, evaluates
                current model. Defaults to False.
        """
        if best:
            self.load_checkpoint(best=True)

        if "val" in split:
            dataloader = get_loader(self.c, "val")
        elif "test" in split:
            dataloader = get_loader(self.c, "test")
        elif "train" in split:
            dataloader = get_loader(self.c, "train")
        else:
            raise ValueError(f"Invalid split passed to evaluation: {split}!")

        with torch.no_grad():

            self.model.eval()
            self.estimator.reset()
            paths = []

            preds = torch.tensor([]).to(self.device)
            events = torch.tensor([]).to(self.device)
            durations = torch.tensor([]).to(self.device)

            for i, dat in enumerate(dataloader):
                if "surv" in self.c.cnn.network.lower():
                    images, labels, e, t, metadata, path = dat
                    e, t = e.to(self.device), t.to(self.device)
                    paths += path

                else:
                    # images, labels, metadata, path = dat # Unused; intended for AMD Grade clf
                    raise NotImplementedError("Only survival training is supported!")

                images = images.to(self.device)

                out_preds, out_activations = self.model(images)

                preds = torch.cat((preds, out_preds))
                events = torch.cat((events, e))
                durations = torch.cat((durations, t))

            survs = self.get_survival_predictions(preds)

            self.estimator.update(survs, events, durations)

            self.model.train()

        self.estimator.add_numpy_preds()

        if log_all:
            self.log_all_performance(split=split)

            if "surv" in self.c.cnn.network.lower():
                # Log survival probabilities for each patient (use image path as id)
                paths = np.array(paths).flatten()
                patient_ids = np.array([os.path.basename(p) for p in paths])
                if not len(patient_ids) == len(self.estimator.survs_np):
                    print("WARNING: patient_ids and survs_np have different lengths!")
                sp = pd.DataFrame(self.estimator.survs_np)
                sp["patient_id"] = patient_ids
                sp.columns = sp.columns.astype(str)
                wandb.log({f"survival_probs/{split}": sp})

    def log_all_performance(self, split: str):
        """Log all performance metrics for a given split. Call evaluate() first!
        Calculates based on previous call of evaluate(). Make sure splits coincide!

        Args:
            split (str): Split to log performance for
        """
        # Evaluate all metrics, e.g. for final evaluation of best model
        performance = self.estimator.get_all_performances()

        # Log for c.num_classes
        self._log(self.c.cnn.num_classes, split, performance)
        # self._log_plots(plots=plots, split=split, num_classes=self.c.cnn.num_classes)

    def train(self):
        """Train model"""

        print("Starting training...")

        if self.c.cnn.test_run["enabled"]:
            print(
                f"Running a test run with a tiny sample of size {self.c.cnn.test_run['size']}!"
            )

        # Progress bar and timer
        pbar_batch = tqdm(
            total=len(self.train_loader) * (self.c.cnn.num_epochs - self.current_epoch),
            desc="Batch",
        )

        self.model.train()

        for epoch in range(self.current_epoch, self.c.cnn.num_epochs):

            if self.warmup_scheduler and not self.warmup_scheduler.finished():
                self.warmup_scheduler.step()

            train_losses = []

            pbar_batch.set_description(f"Epoch {epoch + 1}/{self.c.cnn.num_epochs}")

            self.current_epoch = epoch

            self.estimator.reset()

            preds = torch.tensor([]).to(self.device)
            events = torch.tensor([]).to(self.device)
            durations = torch.tensor([]).to(self.device)

            for i, dat in enumerate(self.train_loader):
                images, labels, e, t, metadata, path = dat
                e, t = e.to(self.device), t.to(self.device)

                self.current_batch += 1

                images = Variable(images.to(self.device))

                out_preds, out_activations = self.model(images)

                # print("out_preds", out_preds)
                # print("shape", out_preds.shape) # 8,1

                preds = torch.cat((preds, out_preds))
                events = torch.cat((events, e))
                durations = torch.cat((durations, t))

                if "surv" in self.c.cnn.network.lower():
                    
                    loss = self.get_survival_loss(
                        out_preds, e, t, out_activations
                    )

                # Get classification-only loss is included in get_survival_loss
                else:
                    raise NotImplementedError("Only survival training is supported!")

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                if self.scheduler and "onecycle" in self.c.cnn.scheduler:
                    if not self.warmup_scheduler or self.warmup_scheduler.finished() and self.warmup_scheduler.epoch != self.current_epoch+1:
                        self.scheduler.step()

                train_losses.append(loss)

                wandb.log(
                    {
                        "learning_rate": get_lr(self.optimizer),
                        "epoch": self.current_epoch,
                        "batch": self.current_batch,
                    }
                )

                pbar_batch.update(1)
            
            if self.scheduler and not "onecycle" in self.c.cnn.scheduler:
                if not self.warmup_scheduler or self.warmup_scheduler and self.warmup_scheduler.finished():
                    self.scheduler.step()
            
            # Preds to survival probabilities using Breslow estimator fit to train set
            #  if not loaded from checkpoint
            if self.type == "cox":
                if self.breslow is None:
                    self.breslow = fit_breslow(preds, t=durations, e=events)

            survs = self.get_survival_predictions(preds)

            self.estimator.update(survs, events, durations)

            pbar_batch.set_description(
                f"Evaluating Epoch {self.current_epoch + 1}/{self.c.cnn.num_epochs}"
            )

            # Log training loss and model selection metric on train set
            model_selection_score_train = self.estimator.get_performance(
                self.c.cnn.model_selection_metric
            )

            self._log_scalar_by_epoch(
                name=f"{self.c.cnn.model_selection_metric}/train",
                value=model_selection_score_train,
                epoch=self.current_epoch,
            )
            self._log_scalar_by_epoch(
                name="loss/train",
                value=torch.stack(train_losses).mean().item(),
                epoch=self.current_epoch,
            )

            # Evaluate on validation set and log model selection metric and loss
            self.evaluate(split="val", best=False, log_all=False)

            model_selection_score = self.estimator.get_performance(
                self.c.cnn.model_selection_metric
            )

            self._log_scalar_by_epoch(
                name=f"{self.c.cnn.model_selection_metric}/val",
                value=model_selection_score,
                epoch=self.current_epoch,
            )

            if self._score_is_better(score=model_selection_score):
                self.best_score = model_selection_score
                self.best_epoch = self.current_epoch
                self.num_bad_epochs = 0
                self.is_best = True

                self._log_scalar_by_epoch(
                    name=f"{self.c.cnn.model_selection_metric}/val_best",
                    value=self.best_score,
                    epoch=self.current_epoch,
                )

            else:
                self.num_bad_epochs += 1
                self.is_best = False
                stopping_criterion = 20
                if hasattr(self.c.cnn, "stop_after_epochs"):
                    if any([arg in str(self.c.cnn.stop_after_epochs).lower() for arg in ["none", "false"]]):
                        stopping_criterion = np.inf
                    else:
                        stopping_criterion = self.c.cnn.stop_after_epochs
                if self.num_bad_epochs >= stopping_criterion:
                    print(
                        f"Stopping training after {self.num_bad_epochs} epochs without improvement."
                    )
                    break

            self.save_checkpoint()

        pbar_batch.close()

        print("Finished training!")

    def _log(self, num_classes: int, split: str, performance: dict):
        """Log performance to wandb. Call evaluate() first!

        Args:
            split (str): Split to log, e.g. "val"
            num_classes (int): Number of classes
            performance: Performance metrics and values to log
        """

        num_classes = int(num_classes)

        for metric in performance.keys():

            if isinstance(performance[metric], list) or isinstance(performance[metric], np.ndarray):
                print(f"logging array {metric}: {performance[metric]}")
                # Log array of performance values by time
                self._log_by_time(
                    name=f"{num_classes}C_{split}/{metric}",
                    values=performance[metric],
                    unit="year",
                )
                # Additionally log the array, as since wandb 0.16.0, _log_by_time does not 
                # overwrite the old values anymore in history (for plots)
                wandb.log(
                    {
                        f"{num_classes}C_{split}/{metric+'_array'}": performance[metric],
                    }
                )
            else:
                # Log scalar value
                wandb.log(
                    {
                        f"{num_classes}C_{split}/{metric}": performance[metric],
                    }
                )

    def _log_scalar_by_epoch(self, name: str, value: float, epoch: int):
        """Log performance of single metric to wandb, irrespective of number of classes.
        Call evaluate() first!

        This is for logging of model selection performance

        Args:
            name (str): Name to log, e.g. "loss/val" (metric/split)
            value (float): Value to log
            epoch (int): Epoch to log
        """
        wandb.log(
            {
                f"{name}": value,
                "epoch": epoch,
            }
        )

    def _log_plots(self, plots: dict, split: str, num_classes: int):
        """Log confusion matrix. Call evaluate() first!"""
        wandb.log({f"{num_classes}C_{split}/confmat_wandb": plots["confmat_wandb"]})

        # Log ROC and PR curves
        wandb.log(
            {
                f"{num_classes}C_{split}/pr_curve": plots["pr_curve"],
            }
        )

        wandb.log(
            {
                f"{num_classes}C_{split}/roc_curve": plots["roc_curve"],
            }
        )

    def _log_by_time(self, name, values, unit="year"):
        """Log an array of performance values of one metric (e.g. dynamic AUC) by evaluation time"""
        evaluation_times = self.c.cnn.survival_times

        if unit == "year":
            # Convert visits to years
            evaluation_times = [t / 2 for t in evaluation_times]
        
        evaluation_times = evaluation_times[:len(values)]

        # For wandb >= 0.16.0:
        wandb.define_metric(unit)
        wandb.define_metric(name, step_metric=unit)

        for i, t in enumerate(evaluation_times):
            wandb.log({f"{name}": values[i], unit: t})