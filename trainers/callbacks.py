import torch

# 定义学习率调整回调函数
class LRSchedulerCallback(object):
    def __init__(self, optimizer, lr_scheduler):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def __call__(self, epoch):
        self.lr_scheduler.step()

# 定义模型保存回调函数
class ModelCheckpointCallback(object):
    def __init__(self, model, save_path):
        self.model = model
        self.save_path = save_path

    def __call__(self, epoch):
        torch.save(self.model.state_dict(), self.save_path)
