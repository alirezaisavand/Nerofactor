import abc
import numpy as np


class LearningRateManager(abc.ABC):
    @staticmethod
    def set_lr_for_all(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def construct_optimizer(self, optimizer, network):
        # may specify different lr for different parts
        # use group to set learning rate
        paras = network.parameters()
        return optimizer(paras, lr=1e-3, eps=1e-4, amsgrad=True)

    @abc.abstractmethod
    def __call__(self, optimizer, step, *args, **kwargs):
        pass


class WarmUpCosLR(LearningRateManager):
    default_cfg = {
        'end_warm': 5000,
        'end_iter': 300000,
        'lr': 5e-4,
    }

    def __init__(self, cfg):
        cfg = {**self.default_cfg, **cfg}
        self.warm_up_end = cfg['end_warm']
        self.learning_rate_alpha = 0.2
        self.end_iter = cfg['end_iter']
        self.learning_rate = cfg['lr']

    def __call__(self, optimizer, step, *args, **kwargs):
        if step < self.warm_up_end:
            learning_factor = step / self.warm_up_end
        else:
            end_iter = 100 * 1000
            nom = min(step, end_iter)
            alpha = self.learning_rate_alpha
            progress = (nom - self.warm_up_end) / (end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        lr = self.learning_rate * learning_factor
        self.set_lr_for_all(optimizer, lr)
        return lr


name2lr_manager = {
    'warm_up_cos': WarmUpCosLR,
}
