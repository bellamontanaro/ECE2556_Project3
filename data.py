import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CovidDataset(Dataset):
    def __init__(self, datas, transform=None):
        self.input_list = datas['imgs']
        self.label_list = datas['masks']
        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        if self.input_list[idx].split('/')[-1] != self.label_list[idx].split('/')[-1]:
            raise ValueError(f'input and label do not match: {self.input_list[idx]} {self.label_list[idx]}')
        input_slice = np.load(self.input_list[idx])
        label_slice = np.load(self.label_list[idx])
        input_slice = Image.fromarray(input_slice)
        label_slice = Image.fromarray(label_slice)

        input_name = self.input_list[idx].split('/')[-1].split('.')[0]

        if self.transform:
            input_slice = self.transform(input_slice)
            label_slice = self.transform(label_slice)

        return input_slice, label_slice, input_name