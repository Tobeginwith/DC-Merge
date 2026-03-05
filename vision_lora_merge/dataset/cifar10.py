import os
import PIL
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from torch.utils.data import random_split
from torchvision.datasets import VisionDataset

cifar_classnames = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
g = torch.Generator().manual_seed(42)

class CIFAR10:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=6,
    ):

        self.train_dataset = PyTorchCIFAR10(
            root=location, download=False, train=True, transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test_dataset = PyTorchCIFAR10(
            root=location, download=False, train=False, transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = self.test_dataset.classes


def convert(x):
    if isinstance(x, np.ndarray):
        return torchvision.transforms.functional.to_pil_image(x)
    return x


class BasicVisionDataset(VisionDataset):
    def __init__(self, images, targets, transform=None, target_transform=None):
        if transform is not None:
            transform.transforms.insert(0, convert)
        super(BasicVisionDataset, self).__init__(
            root=None, transform=transform, target_transform=target_transform
        )
        assert len(images) == len(targets)

        self.images = images
        self.targets = targets

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.targets[index]

    def __len__(self):
        return len(self.targets)


def prepare_train_loaders(config):
    dataset_class = CIFAR10(
        preprocess=config['train_preprocess'],
        location=config['root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    loaders = {
        'full': dataset_class.train_loader
    }
    return loaders


def prepare_test_loaders(config):
    dataset_class = CIFAR10(
        preprocess=config['eval_preprocess'],
        location=config['root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    
    loaders = {'test': dataset_class.test_loader}
    if config.get('val_fraction', 0) > 0.:
        print('splitting cifar10')
        num_valid = int(len(dataset_class.test_dataset) * config['val_fraction'])
        num_test = len(dataset_class.test_dataset) - num_valid
        val_set, test_set = random_split(dataset_class.test_dataset, [num_valid, num_test], generator=g)
        loaders['test'] = torch.utils.data.DataLoader(
            test_set,
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
        loaders['val'] = torch.utils.data.DataLoader(
            val_set, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
    loaders['class_names'] = dataset_class.classnames
    
    return loaders
