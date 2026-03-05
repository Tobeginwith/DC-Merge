import io
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data import random_split
from datasets import load_from_disk

g = torch.Generator().manual_seed(42)

class CustomFER2013Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = Image.open(io.BytesIO(sample["img_bytes"])).convert(
            "L"
        )  # Convert to PIL image
        label = sample["labels"]

        if self.transform:
            image = self.transform(image)

        return image, label


class FER2013:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=6,
    ):

        location = os.path.join(location, "fer2013_dataset")

        # Load the FER2013 dataset using Hugging Face datasets library
        fer2013 = load_from_disk(location)['train']

        # Instantiate the custom PyTorch training dataset
        self.train_dataset = CustomFER2013Dataset(fer2013, transform=preprocess)

        # Use PyTorch DataLoader to create an iterator over training batches
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        # Load the FER2013 test dataset using Hugging Face datasets library
        fer2013_test = load_from_disk(location)['test']

        # Instantiate the custom PyTorch test dataset
        self.test_dataset = CustomFER2013Dataset(fer2013_test, transform=preprocess)

        # Use PyTorch DataLoader to create an iterator over test batches
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = [
            ["angry"],
            ["disgusted"],
            ["fearful"],
            ["happy", "smiling"],
            ["sad", "depressed"],
            ["surprised", "shocked", "spooked"],
            ["neutral", "bored"],
        ]
        
def prepare_train_loaders(config):
    dataset_class = FER2013(
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
    dataset_class = FER2013(
        preprocess=config['eval_preprocess'],
        location=config['root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    
    loaders = {'test': dataset_class.test_loader}
    if config.get('val_fraction', 0) > 0.:
        print('splitting fer2013')
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
