# Includes adaptations from Huang et al., 2022
# https://github.com/YijinHuang/pytorch-classification/blob/master/data/transforms.py

import sys
import os
from types import SimpleNamespace
from typing import List
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, precision_recall_curve, log_loss, roc_curve, roc_auc_score, brier_score_loss


from sksurv.metrics import (
    brier_score,
    integrated_brier_score,
    concordance_index_censored,
    cumulative_dynamic_auc
)

# Import from parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "..")

from data.cnn_surv_dataloader import get_dataset
from utils.cnn_survival_utils import e_t_to_tuple, et_tuple_to_df, get_event_indicator_matrix
from utils.helpers import get_project_dir

try: 
    from statkit.decision import net_benefit
except:
    print("Could not import net_benefit from statkit.decision. Will return NaN for net benefit.")

try: 
    from rpy2.robjects.packages import importr, isinstalled
    from rpy2.robjects import pandas2ri, numpy2ri

    packages = ["survival", "timeROC"]
    for package in packages:
        if not isinstalled(package):
            print("installing ", package)
            rutils = importr("utils")
            rutils.chooseCRANmirror(ind=1)
            rutils.install_packages(package)
            print("done")
except:
    print("Could not import timeROC from R. Will return NaN for PR-Curve/PR-AUC.")

# Whether to evaluate on all passed evaluation times or only on all but the last x ones
TRUNCATE_TIMES_BY = 0


class Metrics:
    """Class for computing metrics for a CNN model after each epoch"""

    def __init__(self, config: SimpleNamespace):
        self.c = config.cnn

        # Load unmodified training data, 
        # needed for computation of ipcw in training set for survival metrics
        cc = deepcopy(config)
        cc.cnn.loss = "any"
        cc.cnn.use_stereo_pairs = False
        y_train_set = get_dataset(split="train", c=cc).get_e_t()

        self.y_train = e_t_to_tuple(
            y_train_set[0],
            y_train_set[1],
        )

        self.reset()

    def reset(self):
        self.survs = []
        self.events = []
        self.times = []

        for attr in ["survs_np", "times_np", "events_np", "y_test_np"]:
            self._reset_attr(attr)

    def _reset_attr(self, attr):
        if hasattr(self, attr):
            delattr(self, attr)

    def print_surv_info(self, message: str = ""):
        self.add_numpy_preds()

        (
            self.truncated_survival_times,
            self.truncated_survs,
        ) = self.truncate_survtimes_preds()

        print("#### INFO ####")
        print(message)
        for var in [
            "truncated_survival_times",
            "truncated_survs",
            "events_np",
            "y_test_np",
        ]:
            print(f"{var} dtype: {type(getattr(self, var))}")
            print(f"{var} values[0]: {getattr(self, var)[0]}")
            print(f"{var} shape: {getattr(self, var).shape}")


    def truncate_survtimes_preds(self, by: int = TRUNCATE_TIMES_BY):
        survival_times = np.array(self.c.survival_times)
        survs = self.survs_np
        if TRUNCATE_TIMES_BY > 0:
            survival_times = np.array(self.c.survival_times)[:-by]
            survs = self.survs_np[:, :-by]
        return survival_times, survs

    def update(self, preds: torch.Tensor, events: torch.Tensor, times: torch.Tensor):
        """
        Updates estimator with current batches predictions and labels.

        Args:
            preds: survival predictions, i.e. probabilities for no event = 1-risk
            events: event indicators
            times: time of events or censoring
        """         
        events, times = events.cpu(), times.cpu()

        # Update lists
        self.survs.append(preds.detach())
        self.events.append(events.detach())
        self.times.append(times.detach())

    def calc_event_indicator_matrix(self):
        """Matrix with 1s at all visits >= duration to event. Times are columns, patients are rows.
        Column index corresponds to time in the unit given by the model's durations/times array.
        
        Example: times = [0, 1, 2, 3, 4, 5]
                 labels are then found through: calc_event_indicator_matrix()[:, times] 
        """
        self.add_numpy_preds()

        return get_event_indicator_matrix(events=self.events_np, durations=self.times_np)
    
    def calc_rocauc_curve_sklearn(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()
        event_indicator_matrix = self.calc_event_indicator_matrix()

        tprs = []
        fprs = []
        aucs = []

        for idx, time in enumerate(survival_times):
            fpr, tpr, thres = roc_curve(
                y_true = event_indicator_matrix[:,time],
                y_score = 1.0 - survs[:,idx],
            )
            tprs.append(tpr)
            fprs.append(fpr)
            aucs.append(auc(fpr, tpr))

        return tprs, fprs, np.array(aucs)

    def calc_precision_recall_curve_sklearn(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()
        event_indicator_matrix = self.calc_event_indicator_matrix()

        auprcs = []
        precisions = []
        recalls = []

        for idx, time in enumerate(survival_times):
            precision, recall, _ = precision_recall_curve(
                y_true = event_indicator_matrix[:,time],
                probas_pred = 1.0 - survs[:,idx],
            )
            auprc = auc(recall, precision)
            auprcs.append(auprc)
            precisions.append(precision)
            recalls.append(recall)

        return precisions, recalls, np.array(auprcs)
    
    def calc_precision_recall_curve_timeroc(self, n_thresholds=1000):
        """ Calculate precision and recall over n_thresholds for each eval_time. Uses the 
            model's survival predictions at the evaluation time! Is censoring adjusted using Kaplan-Meier
            to predict the censoring distribution. Needs R installed and uses packages 
            timeROC and survival.
        """
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        timeROC = importr("timeROC")
        survival = importr("survival")
 
        risks = 1.0 - survs

        cuts = np.arange(0, 1, 1/n_thresholds)

        recalls = []
        precisions = []
        for cid, cut in enumerate(cuts):

            ppv_by_time = []
            tp_by_time = []
            for tid, time in enumerate(survival_times):
                numpy2ri.activate()
                ppv = timeROC.SeSpPPVNPV(
                    cutpoint = cut, 
                    T = self.times_np,
                    delta = self.events_np,
                    marker = risks[:,tid],
                    cause = 1,
                    weighting = "marginal", # KM-IPCW
                    times = time,
                )
                numpy2ri.deactivate()

                # Positive Predictive Value estimates at time. -> Precision
                ppv_by_time.append(ppv.rx2("PPV"))
                # True Positive fraction (sensitivity) estimates at time.
                tp_by_time.append(ppv.rx2("TP")) 

            # timeROC adds a time=0 column at idx=0, ignore it using [:,-1]
            recalls.append(np.array(tp_by_time)[:,-1]) 
            precisions.append(np.array(ppv_by_time)[:,-1])
            
        recalls_ = np.array(recalls)
        precisions_ = np.array(precisions)

        auprcs = []
        precisions = []
        recalls = []
        for tid, time in enumerate(survival_times):
            recall_ = recalls_[:,tid]
            precision_ = precisions_[:,tid]
            precision_[np.isnan(precision_)] = 0

            # Ensure that curve values span the whole range of x axis for AUC calculation
            recall_ = np.append(recall_, 0)
            recall_ = np.insert(recall_, 0, values=1)
            precision_ = np.insert(precision_, 0, values=0)
            precision_ = np.append(precision_, 1)

            precisions.append(precision_)
            recalls.append(recall_)

            auprc = auc(recall_, precision_)
            auprcs.append(auprc)

        self.pr_timeroc = (precisions, recalls, np.array(auprcs))
        
        return precisions, recalls, np.array(auprcs)

    def get_aunbc(self, clip:bool=True):    
        try:
            import statkit
        except:
            return np.nan

        # Area under the net benefit curve
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        y_ = self.calc_event_indicator_matrix()
        risks = 1.0 - np.array(survs)

        net_benefits = {visit: None for visit in survival_times}
        for idx, visit in enumerate(survival_times):
            # Uses risks instead of survival_probs
            net_benefits[visit] = net_benefit(y_[:,visit], risks[:,idx], 
                                    thresholds=np.linspace(0, 1.0, 100), action=True)
            if clip:
                # Get tuple of thresholds and net benefits
                net_benefits[visit] = np.clip(net_benefits[visit], 0, 1)

        aucs = {visit: None for visit in survival_times}
        for idx, visit in enumerate(survival_times):
            aucs[visit] = auc(net_benefits[visit][0], net_benefits[visit][1])

        # To array
        aucs = np.array(list(aucs.values()))

        return aucs

    def get_mean_aunbc(self, clip:bool=True):
        # Mean area under the net benefit curve
        self.add_numpy_preds()

        aucs = self.get_aunbc(clip=clip)

        return np.mean(aucs)


    def get_ibs(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        # Unlike other sksurv metrics, (I)BS takes survs as input instead of risks: 
        # "the estimated probability of remaining event-free up to the i-th time point."
        if len(survival_times) > 1:
            ibs = integrated_brier_score(
                self.y_train, self.y_test_np, survs, survival_times
            )
        else:
            ibs = brier_score(self.y_train, self.y_test_np, survs, survival_times)[1][0]

        return ibs

    def get_ibs_sklearn(self):
        scores = self.get_brier_score_sklearn()
        return np.mean(scores)

    def get_auc(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        # Dynamic AUC (dynamic area under the time-dependent ROC curve) within times[:-1]
        try:
            risks = 1.0 - np.array(survs)
            dyn_auc = cumulative_dynamic_auc(
                self.y_train,
                self.y_test_np,
                risks,
                survival_times,
            )[0]
        except ValueError as e:
            print(f"Error computing AUC: {e}")
            dyn_auc = np.nan

        return dyn_auc
    
    def get_auc_sklearn(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        event_indicator_matrix = self.calc_event_indicator_matrix()
        aucs = []
        for idx, time in enumerate(survival_times):
            aucs.append(roc_auc_score(
                y_true = event_indicator_matrix[:,time],
                y_score = 1.0 - survs[:,idx],
            ))
        
        return np.array(aucs)
    
    def get_mean_auc(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        # Dynamic AUC (dynamic area under the time-dependent ROC curve) within times[:-1]
        try:
            risks = 1.0 - np.array(survs)
            mean_auc = cumulative_dynamic_auc(
                self.y_train,
                self.y_test_np,
                risks,
                survival_times,
            )[1]
        except ValueError as e:
            print(f"Error computing Mean AUC: {e}")
            mean_auc = np.nan

        return mean_auc

    def get_mean_auc_sklearn(self):
        aucs = self.get_auc_sklearn()
        return np.mean(aucs)

    def get_auprc(self, method="timeroc"):
        """ Get area under the precision-recall curve. 

        Args:
            method (str): If "timeroc", uses Rs timeROC package and computes the AUPRC for each 
                evaluation time using ipcw-adjusted sensitivity and precision estimates. It makes
                use of the model's survival predictions at the eval_time.
                If "sklearn", uses sklearn's precision_recall_curve function to compute the AUPRC,
                i.e. does not adjust for censoring. Also uses the model's survival predictions at
                the eval_time.
        
        Returns:
            auprcs (np.array): AUPRC at each evaluation time
        """
        try:
            import rpy2
        except:
            return np.nan

        if method =="timeroc":
            if hasattr(self, "pr_timeroc"):
                return self.pr_timeroc[2]
            else:
                return self.calc_precision_recall_curve_timeroc()[2]
        elif method == "sklearn":
            return self.calc_precision_recall_curve_sklearn()[2]
        else:
            raise ValueError(f"Invalid method: {method}. Valid methods are: 'timeroc', 'sklearn'")
    
    def get_auprc_sklearn(self):
        return self.get_auprc(method="sklearn")
    
    def get_mean_auprc(self, method="timeroc"):
        if method == "timeroc":
            try:
                import rpy2
            except:
                return np.nan
            if hasattr(self, "pr_timeroc"):
                return np.mean(self.pr_timeroc[2])
            else:
                return np.mean(self.calc_precision_recall_curve_timeroc()[2])
        elif method == "sklearn":
            return np.mean(self.calc_precision_recall_curve_sklearn()[2])
        else:
            return np.nan

    def get_mean_auprc_sklearn(self):
        return self.get_mean_auprc(method="sklearn")
    
    def get_brier_score(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()
        
        # BS takes survs as input: 
        # "the estimated probability of remaining event-free up to the i-th time point."
        return brier_score(self.y_train, self.y_test_np, survs, survival_times)[1]
    
    def get_brier_score_sklearn(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        event_indicator_matrix = self.calc_event_indicator_matrix()
        scores = []
        for idx, time in enumerate(survival_times):
            scores.append(brier_score_loss(
                y_true = event_indicator_matrix[:,time],
                y_prob = 1.0 - survs[:,idx],
            ))
        
        return np.array(scores)
    
    def get_concordance_index(self):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        # Harrell's C-index over all times
        # Note: This has serious issues when using non-proportional models as risk curves between
        # patients can cross. As discussed here (https://github.com/havakv/pycox/issues/33)
        # it is suggested to either use Antolini's C instead
        # (https://github.com/havakv/pycox/blob/master/pycox/evaluation/concordance.py),
        # or integrate/sum over all risks over times by patient. I do the latter here.
        # Idea: If our model is any good, a patient with shorter time-to-event will have a 
        # higher sum of risks than a patient with higher time-to-event.

        # Get sum of risks over all times for each patient
        risks = 1.0 - np.array(survs)
        sum_risk_c = np.sum(risks, axis=1) if len(survival_times) > 1 else risks.flatten()
        # Sort by durations of patients to evaluate
        _y = et_tuple_to_df(self.y_test_np)
        _durations = _y["time"]
        _events = _y["event"]
        idx = pd.Series(_durations).sort_values(ascending=False).index
        durations_c = _durations[idx]
        events_c = _events[idx]
        sum_risk_c = sum_risk_c[idx]
        c_harrell = concordance_index_censored(events_c, durations_c, sum_risk_c)[0]

        return c_harrell
    
    def get_bce_loss(self):
        """Binary cross entropy loss (neg. log likelihood) at the evaluation times specified in the
        config. 
        
        Using sklearn's log_loss or torch.nn.BCELoss(reduction="none") gives the same results
        for a single time point, i.e. when evaluating (a classification model) at one time point 
        only. However, when evalauting at multiple time points, they differ. Sticking to torch
        as it explicitly allows for the input and target shapes."""        
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        event_indicator_matrix = self.calc_event_indicator_matrix()
        bce = torch.nn.functional.binary_cross_entropy
        logloss = bce(
            torch.tensor(1.0-survs).float(), 
            torch.tensor(event_indicator_matrix[:,survival_times]).float()
        ).item()

        return logloss

    def get_all_performances(self):
        self.add_numpy_preds()
        print("Set length: ", len(self.y_test_np))

        out = {}
        for metric in self.get_valid_metrics():
            out[metric] = self.get_performance(metric)

        return out

    def get_performance(self, metric: str):

        try:
            out = getattr(self, f"get_{metric}")()
        except AttributeError as e:
            print(e)
            raise ValueError(
                f"Invalid metric: {metric}. Valid metrics are: {self.get_valid_metrics()}"
            )

        return out

    def get_valid_metrics(self):
        prefix = "get_"
        valid_metrics = [
            metric[len(prefix) :] for metric in dir(self) if metric.startswith("get_")
        ]
        valid_metrics = [
            metric
            for metric in valid_metrics
            if not "metrics" in metric
            and not "labels" in metric
            and not "confmat" in metric
            and not "performance" in metric
        ]

        return valid_metrics

    def _add_numpy_survs(self):
        if not hasattr(self, "survs_np"):
            self.survs_np = torch.cat(self.survs, dim=0).numpy()
            self.times_np = torch.cat(self.times, dim=0).numpy().astype(int)
            self.events_np = torch.cat(self.events, dim=0).numpy().astype(int)
            self.y_test_np = e_t_to_tuple(self.events_np, self.times_np)

    def add_numpy_preds(self):
        # survival predictions
        self._add_numpy_survs()

    def plot_roccurve(self, method="sklearn", return_values = False):
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        tprs, fprs, aucs = self.calc_rocauc_curve_sklearn()

        fig, ax = plt.subplots(figsize=(6,6))
        for idx, visit in enumerate(survival_times):
            plt.plot(fprs[idx], tprs[idx], label=f"Year: {int(visit/2)}, AUC: {aucs[idx]:.3f}")
        
        ax.set_xlabel("TPR")
        ax.set_ylabel("FPR")
        ax.set_ylim(0, 1.005)
        ax.set_xlim(0, 1.005)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(loc="lower left", frameon=False)

        if return_values:
            return fig, ax, tprs, fprs, aucs
        else:
            return fig, ax

    def plot_prcurve(self, method="timeroc", n_thresholds=1000, return_values=False):
        try:
            import rpy2
        except:
            return np.nan
        
        # reset plt 
        try:
            plt.close()
        except:
            pass
        
        self.add_numpy_preds()

        survival_times, survs = self.truncate_survtimes_preds()

        if method == "timeroc":
            if hasattr(self, "pr_timeroc"):
                precisions, recalls, auprcs = self.pr_timeroc
            else:
                precisions, recalls, auprcs = self.calc_precision_recall_curve_timeroc(n_thresholds)
        elif method == "sklearn":
            precisions, recalls, auprcs = self.calc_precision_recall_curve_sklearn()
        else:
            raise ValueError(f"Invalid method: {method}. Valid methods are: 'timeroc', 'sklearn'")

        fig, ax = plt.subplots(figsize=(6, 6))
        for idx, visit in enumerate(survival_times):
            plt.plot(recalls[idx], precisions[idx], label=f"Year: {int(visit/2)}, AUC: {auprcs[idx]:.3f}")
        
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_ylim(0, 1.005)
        ax.set_xlim(0, 1.005)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(loc="lower left", frameon=False)

        if return_values:
            return fig, ax, precisions, recalls, auprcs
        else:
            return fig, ax
    
    def save_roccurve(self, path: str="", name:str="roccurve", method="sklearn", save_values=False):
        if hasattr(self.c, "run_id"):
            name = f"{self.c.run_id}_{name}"

        if not path.startswith("/"):
            path = os.path.join(get_project_dir(), path)

        fig, ax, tprs, fprs, aucs = self.plot_roccurve(method, return_values = True)
        os.makedirs(path, exist_ok=True)
        for filetype in ["pdf", "png"]:
            fig.savefig(os.path.join(path, f"{name}_{method}.{filetype}"))
        
        if save_values:
            # Save values for plot reproduction
            df = pd.DataFrame(tprs)
            df.to_csv(os.path.join(path, f"{name}_{method}_tprs.csv"), index=False)
            df = pd.DataFrame(fprs)
            df.to_csv(os.path.join(path, f"{name}_{method}_fprs.csv"), index=False)

    def save_prcurve(self, path: str="", name:str="prcurve", method="timeroc", save_values=False):
        if hasattr(self.c, "run_id"):
            name = f"{self.c.run_id}_{name}"

        if not path.startswith("/"):
            path = os.path.join(get_project_dir(), path)

        fig, ax, precisions, recalls, auprcs = self.plot_prcurve(method, return_values = True)
        os.makedirs(path, exist_ok=True)
        for filetype in ["pdf", "png"]:
            fig.savefig(os.path.join(path, f"{name}_{method}.{filetype}"))
        
        if save_values:
            # Save values for plot reproduction
            df = pd.DataFrame(precisions)
            df.to_csv(os.path.join(path, f"{name}_{method}_precisions.csv"), index=False)
            df = pd.DataFrame(recalls)
            df.to_csv(os.path.join(path, f"{name}_{method}_recalls.csv"), index=False)

        
        

            

        

