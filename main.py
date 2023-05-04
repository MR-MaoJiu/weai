import torch
from data.dataset import MyDataset
from torch.utils.data import DataLoader

from models.model import MyModel, criterion, metrics
from trainers.trainer import Trainer
from trainers.optimizer import get_optimizer
from utils.utils import setup_logger, save_model, load_model

# 定义训练和测试数据集
train_dataset = MyDataset('data/train_data', 'data/train_labels')
test_dataset = MyDataset('data/test_data', 'data/test_labels')

# 定义训练和测试数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义模型
model = MyModel().to(device)

# 定义优化器
optimizer = get_optimizer(model)

# 定义训练器
trainer = Trainer(model, train_loader, test_loader, optimizer, criterion, metrics, device)

# 定义日志管理器
logger = setup_logger('my_logger', 'logs/train.log')

# 训练模型
trainer.train(epochs=10)

# 保存模型
save_model(model, 'models/my_model.pth')

# 加载模型
model = load_model(model, 'models/my_model.pth')
