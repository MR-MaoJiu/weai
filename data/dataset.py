
from PIL import Image
from torch.utils import data

from data.preprocess import preprocess
from utils.utils import load_data, load_labels


class MyDataset(data.Dataset):
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
