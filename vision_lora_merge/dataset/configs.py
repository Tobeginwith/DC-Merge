from .eurosat import EuroSATBase
from .cars import Cars
from .dtd import DTD
from .mnist import MNIST
from .gtsrb import GTSRB
from .svhn import SVHN
from .sun397 import SUN397
from .resisc45 import RESISC45
from .cifar100 import CIFAR100
from .flowers102 import Flowers102
from .oxfordpets import OxfordIIITPet
from .stl10 import STL10
from .sst2 import RenderedSST2
from .fashionmnist import FashionMNIST
from .fer2013 import FER2013
from .cifar10 import CIFAR10


BASE_DIR = ''


eurosat = {
    'wrapper': EuroSATBase,
    'batch_size': 128,
    'res': 224,
    'type': 'eurosat',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/eurosat',
    'root': BASE_DIR
}

stanford_cars = {
    'wrapper': Cars,
    'batch_size': 128,
    'res': 224,
    'type': 'stanford_cars',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/stanford_cars',
    'root': BASE_DIR
}

mnist = {
    'wrapper': MNIST,
    'batch_size': 128,
    'res': 224,
    'type': 'mnist',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/mnist',
    'root': BASE_DIR
}

svhn = {
    'wrapper': SVHN,
    'batch_size': 128,
    'res': 224,
    'type': 'svhn',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/svhn',
    'root': BASE_DIR
}

dtd = {
    'wrapper': DTD,
    'batch_size': 128,
    'res': 224,
    'type': 'dtd',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/dtd',
    'root': BASE_DIR
}

sun397 = {
    'wrapper': SUN397,
    'batch_size': 128,
    'res': 224,
    'type': 'sun397',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/SUN397',
    'root': BASE_DIR
}

gtsrb = {
    'wrapper': GTSRB,
    'batch_size': 128,
    'res': 224,
    'type': 'gtsrb',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/gtsrb',
    'root': BASE_DIR
}

resisc45 = {
    'wrapper': RESISC45,
    'batch_size': 128,
    'res': 224,
    'type': 'resisc45',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/resisc45',
    'root': BASE_DIR
}

cifar100 = {
    'wrapper': CIFAR100,
    'batch_size': 128,
    'res': 224,
    'type': 'cifar100',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/cifar100',
    'root': BASE_DIR
}

flowers102 = {
    'wrapper': Flowers102,
    'batch_size': 128,
    'res': 224,
    'type': 'flowers102',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/flowers102',
    'root': BASE_DIR
}

oxfordpets = {
    'wrapper': OxfordIIITPet,
    'batch_size': 128,
    'res': 224,
    'type': 'oxfordpets',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/oxfordpets',
    'root': BASE_DIR
}

stl10 = {
    'wrapper': STL10,
    'batch_size': 128,
    'res': 224,
    'type': 'stl10',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/stl10',
    'root': BASE_DIR
}

sst2 = {
    'wrapper': RenderedSST2,
    'batch_size': 128,
    'res': 224,
    'type': 'sst2',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/sst2',
    'root': BASE_DIR
}

fashionmnist = {
    'wrapper': FashionMNIST,
    'batch_size': 128,
    'res': 224,
    'type': 'fashionmnist',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/fashionmnist',
    'root': BASE_DIR
}

fer2013 = {
    'wrapper': FER2013,
    'batch_size': 128,
    'res': 224,
    'type': 'fer2013',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/fer2013',
    'root': BASE_DIR
}

cifar10 = {
    'wrapper': CIFAR10,
    'batch_size': 128,
    'res': 224,
    'type': 'cifar10',
    'num_workers': 8,
    'shuffle_train': True,
    'shuffle_test': False,
    'dir': './data/cifar10',
    'root': BASE_DIR
}


