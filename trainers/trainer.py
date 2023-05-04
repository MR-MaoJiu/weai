import torch.optim as optim
from opt_einsum.backends import torch
from torch.utils.data import DataLoader

# 定义训练函数
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        train_acc += (outputs.argmax(1) == labels).sum().item()
    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)
    return train_loss, train_acc

# 定义测试函数
def test(model, test_loader, criterion, metrics, device):
    model.eval()
    test_loss = 0.0
    test_metrics = {k: 0.0 for k in metrics}
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            for k, metric in metrics.items():
                test_metrics[k] += metric(outputs, labels) * inputs.size(0)
    test_loss /= len(test_loader.dataset)
    test_metrics = {k: v / len(test_loader.dataset) for k, v in test_metrics.items()}
    return test_loss, test_metrics

# 定义训练器类
class Trainer(object):
    def __init__(self, model, train_loader, test_loader, optimizer, criterion, metrics, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.device = device

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = train(self.model, self.train_loader, self.optimizer, self.criterion, self.device)
            test_loss, test_metrics = test(self.model, self.test_loader, self.criterion, self.metrics, self.device)
            print('Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch+1, train_loss, train_acc, test_loss, test_metrics['accuracy']))
