import os

VIT_ARCH = 'ViT-L-14-CLIP-new'  # Model Architecture
MODEL_DIR = ''              # Model Directory
CACHE_DIR = ''              # Where to cache HF pretrained checkpoints
HEAD_DIR = ''               # CLIP Head Directory
FT_DIR = ''                 # Fine-tuned model directory

config = {
    'dataset': [
        {
            'name': 'stanford_cars',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'stanford_cars_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/cars_shuffled_idxs.pt')
        },
        {
            'name': 'dtd',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'dtd_head.pt'),
            'batch_size': 32,
            'num_workers': 16,
        },
        {
            'name': 'eurosat',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'eurosat_head.pt'),
            'batch_size': 32,
            'num_workers': 16,
        },
        {
            'name': 'gtsrb',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'gtsrb_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/gtsrb_shuffled_idxs.pt')
        },
        {
            'name': 'mnist',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'mnist_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,  
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/mnist_shuffled_idxs.pt')
        },
        {
            'name': 'resisc45',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'resisc45_head.pt'),
            'batch_size': 32,
            'num_workers': 16,
        },
        {
            'name': 'sun397',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'sun397_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/sun397_shuffled_idxs.pt')
        },
        {
            'name': 'svhn',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'svhn_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/svhn_shuffled_idxs.pt')
        },
        {
            'name': 'cifar100',
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'cifar100_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,
        },
        {
            'name': 'flowers102',
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'flowers102_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,
        },
        {
            'name': 'oxfordpets',
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'oxfordpets_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,
        },
        {
            'name': 'stl10',
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'stl10_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,
        },
    ],
    'model': {
        'name': 'hf_clip',
        'base_type': "ViT-L-14",
        'cachedir': CACHE_DIR,
        'bases': [
            # Path to model checkpoints stored locally - rank-16 12 vision tasks
            f'{FT_DIR}/stanford_cars',
            f'{FT_DIR}/dtd',
            f'{FT_DIR}/eurosat',
            f'{FT_DIR}/gtsrb',
            f'{FT_DIR}/mnist',
            f'{FT_DIR}/resisc45',
            f'{FT_DIR}/sun397',
            f'{FT_DIR}/svhn',
            f'{FT_DIR}/cifar100',
            f'{FT_DIR}/flowers102',
            f'{FT_DIR}/oxfordpets',
            f'{FT_DIR}/stl10',
        ],
        'ft_config': {
            'type': 'lora',
            'r': 16,
            'lora_alpha': 16,
            'target_modules': ["q_proj", "k_proj", "v_proj", "out_proj"],
            'lora_dropout': 0.1,
            'bias': "none",
        },
    },
    'eval_type': 'clip'
}

