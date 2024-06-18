from pprint import pprint
import wandb

from model.metrics import Metrics
from model.main import CNN
from utils.helpers import get_config

c = None

def run_sweep():
    wandb_run = wandb.init()

    global c

    # Get settings from sweep
    for key, value in wandb_run.config.items():
        setattr(c.cnn, key, value)

    print("Running sweep agent with config")
    pprint(c)

    run()

def run():
    global c

    estimator = Metrics(c)
    cnn = CNN(c=c, estimator=estimator)

    cnn.train()
    print(f"{c.cnn.model_selection_metric}", cnn.estimator.get_performance(c.cnn.model_selection_metric))

    for split in get_eval_splits(c):
        print(f"Evaluating {split} set...")
        cnn.evaluate(split=split)
        
    print("done")

def get_eval_splits(c):
    if hasattr(c.cnn, "eval_sets"):
        splits = c.cnn.eval_sets
        if not isinstance(splits, list):
            splits = [splits]
    else: 
        splits = ["val"]
    
    return splits

if __name__ == "__main__":
    from utils.helpers import Parser

    parser = Parser()

    parser.add_argument("--config", type=str)
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--sweep_count", type=int, default=1)
    args = parser.parse_args()

    c = get_config(args.config)

    if args.sweep_id not in ["None", "none", None]:
        wandb.agent(args.sweep_id, function=run_sweep, count=args.sweep_count, project=c.cnn.project)
    else:
        run()



    


