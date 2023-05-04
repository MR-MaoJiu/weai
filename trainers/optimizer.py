import torch.optim as optim

# 定义优化器类型和参数
optimizer_type = 'Adam'
optimizer_params = {'lr': 0.001, 'weight_decay': 0.0001}

# 定义获取优化器函数
def get_optimizer(model):
    optimizer = getattr(optim, optimizer_type)(model.parameters(), **optimizer_params)
    return optimizer
