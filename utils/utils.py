import logging
import os
from PIL import Image
from opt_einsum.backends import torch


# 定义日志管理函数
def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

# 定义模型保存函数
def save_model(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)

# 定义模型加载函数
def load_model(model, load_path):
    model.load_state_dict(torch.load(load_path))
    return model
# 返回标签列表
def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = [int(line.strip()) for line in f]
    return labels

# 返回图像数据列表
def load_data(data_path):
    data = []
    for file_name in os.listdir(data_path):
        if file_name.endswith('.jpg'):
            file_path = os.path.join(data_path, file_name)
            image = Image.open(file_path)
            data.append(image)
    return data
