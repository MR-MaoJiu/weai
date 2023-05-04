from tkinter import Image

import torchvision.transforms as transforms
from opt_einsum.backends import torch

from utils.utils import load_data, load_labels


# 定义数据预处理函数
def preprocess(image):
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


# 定义数据集类
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path):
        self.data = load_data(data_path)
        self.labels = load_labels(label_path)

    def __getitem__(self, index):
        image = Image.open(self.data[index])
        label = self.labels[index]
        image = preprocess(image)
        return image, label

    def __len__(self):
        return len(self.data)
