import numpy as np
import torch
from torch.utils.data import Dataset

import boto3


class DigitDataset(Dataset):
    def __init__(self, images=None, labels=None):
        self.images = [] if not images else images
        self.labels = [] if not labels else labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def add_item(self, image, label):
        self.images.append(image)
        self.labels.append(label)

    def shuffle(self):
        permutation = np.random.permutation(len(self.labels))
        new_images = []
        new_labels = []
        for p in permutation:
            new_images.append(self.images[p])
            new_labels.append(self.labels[p])
        self.images = new_images
        self.labels = new_labels

    def save(self, path):
        """Save dataset in compressed format"""
        torch.save({
            'images': self.images,
            'labels': self.labels
        }, path, _use_new_zipfile_serialization=True)
    
    @classmethod
    def load(cls, path, download: str = None, bucket_name: str = ''):
        """Load dataset from compressed file"""
        if download:
            s3_client = boto3.client('s3')
            print(f'Loading data from {bucket_name}/{download} to {path}')
            s3_client.download_file(bucket_name, download, path)
        data = torch.load(path, weights_only=False)
        return cls(data['images'], data['labels'])
