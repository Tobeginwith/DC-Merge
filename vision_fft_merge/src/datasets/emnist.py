import os

import torch

import torchvision
import torchvision.datasets as datasets


def rotate_img(img):
    return torchvision.transforms.functional.rotate(img, -90)


def flip_img(img):
    return torchvision.transforms.functional.hflip(img)


def emnist_preprocess():
    return torchvision.transforms.Compose(
        [
            rotate_img,
            flip_img,
        ]
    )


class EMNIST:
    def __init__(
        self,
        preprocess,
        location,
        batch_size=128,
        num_workers=8,
    ):
        preprocess1 = emnist_preprocess()
        preprocess = torchvision.transforms.Compose(
            [
                preprocess,
                preprocess1,
            ]
        )
        # location = os.path.join(location, "EMNIST")
        self.train_dataset = datasets.EMNIST(
            root=location,
            download=False,
            split="digits",
            transform=preprocess,
            train=True,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.EMNIST(
            root=location,
            download=False,
            split="digits",
            transform=preprocess,
            train=False,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = self.train_dataset.classes
