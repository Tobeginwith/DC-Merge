import os
import torch
from torch.utils.data import random_split
import torchvision.datasets as datasets

g = torch.Generator().manual_seed(42)

class STL10:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=6,
    ):

        # location = os.path.join(location, "stl10")
        self.train_dataset = datasets.STL10(
            root=location, download=False, split="train", transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.STL10(
            root=location, download=False, split="test", transform=preprocess
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = self.train_dataset.classes

def prepare_train_loaders(config):
    dataset_class = STL10(
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
    dataset_class = STL10(
        preprocess=config['eval_preprocess'],
        location=config['root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    
    loaders = {'test': dataset_class.test_loader}
    if config.get('val_fraction', 0) > 0.:
        print('splitting stl10')
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

