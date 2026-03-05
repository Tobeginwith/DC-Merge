# [CVPR 2026] DC-Merge: Improving Model Merging with Directional Consistency
This is the official implementation of our CVPR 2026 paper **DC-Merge: Improving Model Merging with Directional Consistency.**

## Merging Vision Models
### Environment

```bash
conda env create -f environment_vision.yml
conda activate dcmerge_vision
# cd ./vision_fft_merge     FFT merging
cd ./vision_lora_merge    # LoRA merging
```

### Hardware
The experiments can be reproduced using a single NVIDIA 4090 GPU with 24GB of memory.

### Datasets
The datasets are structured as follows. For FFT merging, update the `data_location` in [config.yaml](vision_fft_merge/config/config.yaml) before running the code. For LoRA merging, update the `BASE_DIR` in [configs.py](vision_lora_merge/dataset/configs.py) before running the code.

```sh
/your_dataset_path/MTIL
 ├─ cifar-100-python
 │  ├─ meta
 │  ├─ test
 │  └─ train
 ├─ dtd
 │  ├─ train
 │  ├─ val
 │  └─ test
 ├─ eurosat
 │  ├─ train
 │  ├─ val
 │  └─ test
 ├─ flowers-102
 │  ├─ jpg
 │  ├─ 102flowers.tgz
 │  ├─ imagelabels.mat
 │  └─ setid.mat
 ├─ food-101
 │  ├─ images
 │  └─ meta
 ├─ MNIST/raw
 │  ├─ t10k-images-idx3-ubyte
 │  ├─ t10k-labels-idx1-ubyte
 │  ├─ train-images-idx3-ubyte
 │  └─ train-labels-idx1-ubyte
 ├─ FashionMNIST/raw
 │  ├─ t10k-images-idx3-ubyte
 │  ├─ t10k-labels-idx1-ubyte
 │  ├─ train-images-idx3-ubyte
 │  └─ train-labels-idx1-ubyte
 ├─ KMNIST/raw
 │  ├─ t10k-images-idx3-ubyte
 │  ├─ t10k-labels-idx1-ubyte
 │  ├─ train-images-idx3-ubyte
 │  └─ train-labels-idx1-ubyte
 ├─ oxford-iiit-pet
 │  ├─ annotations
 │  └─ images
 ├─ cars
 │  ├─ cars_test
 │  ├─ cars_train
 │  ├─ devkit
 │  └─ cars_test_annos_withlabels.mat
 └─ sun397
    ├─ train
       ├─ a_abbey
       ├─ a_airplane_cabin
       └─ ...
    └─ val
```

### Checkpoints
The checkpoints we used for Table 1 (LoRA merging) are provided [in this link](https://drive.google.com/drive/folders/13-X9wjnHc4zSkQuZqcfEtVnKVsZItYYP?usp=drive_link).

The checkpoints we used for Table 2 (FFT merging) are provided [in this link](https://drive.google.com/drive/folders/1UEM1Thcz1c7dc1nji1i5uTN53Kf6G3-e).

For FFT merging, the pretrained ViT models are automatically downloaded when running the code.

## Merging Vision-Language Models


## Acknowledgements


## Citation
If you find this repository useful for your work, please consider citing our paper:

