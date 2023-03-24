from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class CDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None,):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path,)
        label = self.img_labels.iloc[idx, -1]
        if self.transform:
            image = self.transform(image)
        return image, label


def train_val_split(dataloader, batch_size, val_percent=0.1, shuffle = True):
    dataset_size = len(dataloader)
    indices = list(range(dataset_size))
    split = int(np.floor(val_percent * dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(dataloader, batch_size=batch_size,
                                                  sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataloader, batch_size=batch_size,
                                                sampler=val_sampler)
    return train_loader, val_loader


def calc_metrics(outputs, labels):
    "calculates accuracy, precision, recall, f1 score, and confusion matrix"
    acc = (outputs == labels).float().mean()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, outputs, average='macro')
    cm = confusion_matrix(labels, outputs)
    return acc, precision, recall, f1, cm