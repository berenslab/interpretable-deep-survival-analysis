import torch

class WarmupLRScheduler():
    def __init__(self, optimizer, warmup_epochs, initial_lr):
        self.epoch = 0
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr

    def step(self):
        if self.epoch <= self.warmup_epochs:
            self.epoch += 1
            curr_lr = (self.epoch / self.warmup_epochs) * self.initial_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = curr_lr

    def finished(self):
        return self.epoch >= self.warmup_epochs

def create_scheduler(cnn):
    if cnn.c.cnn.scheduler is None or "none" in cnn.c.cnn.scheduler.lower():
            return None, None
    if "onecycle" in cnn.c.cnn.scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            cnn.optimizer,
            max_lr=cnn.c.cnn.lr,
            epochs=cnn.c.cnn.num_epochs,
            steps_per_epoch=len(cnn.train_loader),
        )

    elif "cosine" in cnn.c.cnn.scheduler and not "restart" in cnn.c.cnn.scheduler:
        if hasattr(cnn.c.cnn, "scheduler_cosine_len"):
            period = cnn.c.cnn.scheduler_cosine_len
        else:
            period = cnn.c.cnn.num_epochs/2
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            cnn.optimizer, T_max=int(period)
        )

    elif "cosine" in cnn.c.cnn.scheduler and "restart" in cnn.c.cnn.scheduler:
        if hasattr(cnn.c.cnn, "scheduler_cosine_len"):
            length = cnn.c.cnn.scheduler_cosine_len
        else:
            length = cnn.c.cnn.num_epochs/2
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            cnn.optimizer, T_0=length, T_mult=1
        )
        
    else:
        raise ValueError(f"Invalid scheduler: {cnn.c.cnn.scheduler}!")

    if hasattr(cnn.c.cnn, "warmup_epochs") and cnn.c.cnn.warmup_epochs > 0:
        warmup_scheduler = WarmupLRScheduler(
            cnn.optimizer, cnn.c.cnn.warmup_epochs, float(cnn.c.cnn.lr)/10
        )
        return scheduler, warmup_scheduler
    else:
        return scheduler, None