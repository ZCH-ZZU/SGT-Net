"""Learning-rate schedulers with optional linear warm-up."""

from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """Linearly warm up the learning rate, then hand over to a base scheduler."""

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return [group['lr'] for group in self.optimizer.param_groups]
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch)
                    for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs]

    def step(self, metrics=None, epoch=None):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if isinstance(self.after_scheduler, ReduceLROnPlateau):
                    self.after_scheduler.step(metrics)
                else:
                    self.after_scheduler.step()
        else:
            super().step()


def get_lr_scheduler(optimizer, config, total_epochs):
    if not config or not config.get('enabled', False):
        return None

    name = config.get('name', 'ReduceLROnPlateau')
    params = config.get('params', {})
    print(f'      -> Initializing LR Scheduler: {name}')

    if name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=params.get('mode', 'min'),
            factor=params.get('factor', 0.1), patience=params.get('patience', 10),
            verbose=params.get('verbose', True), min_lr=params.get('min_lr', 0))
    elif name == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=params.get('step_size', 30),
            gamma=params.get('gamma', 0.1), verbose=params.get('verbose', True))
    elif name == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=params.get('T_max', total_epochs),
            eta_min=params.get('eta_min', 0), verbose=params.get('verbose', True))
    else:
        raise ValueError(f'Unsupported scheduler name: {name}')

    warmup = config.get('warmup', {})
    if warmup.get('enabled', False):
        print(f"        ... with Gradual Warmup for {warmup.get('epochs', 0)} epochs.")
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=warmup.get('multiplier', 1.0),
            total_epoch=warmup.get('epochs', 0), after_scheduler=scheduler)
    return scheduler
