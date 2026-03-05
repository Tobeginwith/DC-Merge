# [CVPR 2026] DC-Merge: Improving Model Merging with Directional Consistency
This is the official implementation of our CVPR 2026 paper **DC-Merge: Improving Model Merging with Directional Consistency.**

## Merging Vision Models
### Environment
Create an environment and install dependencies:
```bash
conda env create -f environment_vision.yml
conda activate dcmerge_vision
# cd ./vision_fft_merge     FFT merging
cd ./vision_lora_merge    # LoRA merging
```

### Hardware
The experiments can be reproduced using a single NVIDIA 4090 GPU with 24GB of memory.

### Datasets
The datasets are structured as follows. For FFT merging, specify the `data_location` with `/your_dataset_path` in [config.yaml](vision_fft_merge/config/config.yaml) before running the code. For LoRA merging, specify the `BASE_DIR` with `/your_dataset_path` in [configs.py](vision_lora_merge/dataset/configs.py) before running the code.

```sh
/your_dataset_path
 ├─ cifar-10-batches-py
 │  ├─ data_batch_1
 │  ├─ data_batch_2
 │  └─ ...
 ├─ resisc45
 │  ├─ airplane
 │  ├─ airport
 │  ├─ ...
 │  ├─ wetland
 │  ├─ resisc45-train.txt
 │  ├─ resisc45-val.txt
 │  └─ resisc45-test.txt
 ├─ pcam
 │  ├─ camelyonpatch_level_2_split_train_x.h5
 │  ├─ camelyonpatch_level_2_split_train_y.h5
 │  ├─ camelyonpatch_level_2_split_test_x.h5
 │  └─ camelyonpatch_level_2_split_test_y.h5
 ├─ rendered-sst2
 │  ├─ train
      ├─ negative
      └─ positive
 │  ├─ valid
 │  └─ test
 ├─ stl10-binary
 │  ├─ train_X.bin
 │  ├─ train_y.bin
 │  ├─ test_X.bin
 │  ├─ test_y.bin
 │  ├─ class_names.txt
 │  ├─ fold_indices.txt
 │  └─ unlabeled_X.bin
 ├─ cifar-100-python
 │  ├─ meta
 │  ├─ test
 │  └─ train
 ├─ dtd
 │  ├─ train
       ├─ banded
       ├─ blotchy
       └─ ...
 │  ├─ val
 │  └─ test
 ├─ gtsrb
 │  ├─ GTSRB
       ├─ Training
       └─ Final_test
 │  └─ GT-final_test.csv
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
 ├─ fer2013_dataset
 │  ├─ train
       ├─ data-00000-of-00001.arrow
       └─ state.json
 │  └─ test
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
 ├─ EMNIST/raw
 │  ├─ emnist-digits-train-labels-idx1-ubyte
 │  ├─ emnist-digits-test-labels-idx1-ubyte
 │  ├─ emnist-digits-train-images-idx3-ubyte
 │  └─ emnist-digits-test-images-idx3-ubyte
 ├─ oxford-iiit-pet
 │  ├─ annotations
 │  └─ images
 ├─ cars
 │  ├─ cars_test
 │  ├─ cars_train
 │  ├─ devkit
 │  └─ cars_test_annos_withlabels.mat
 ├─ svhn
 │  ├─ train_32x32.mat
 │  └─ test_32x32.mat
 └─ sun397
    ├─ train
       ├─ a_abbey
       ├─ a_airplane_cabin
       └─ ...
    └─ val
```

### Checkpoints
The checkpoints we used for Table 1 (LoRA merging) are provided [in this link](https://drive.google.com/drive/folders/13-X9wjnHc4zSkQuZqcfEtVnKVsZItYYP?usp=drive_link). Remember to specify the `FT_DIR` with `/your_model_path/lora_checkpoints` in scripts under the [configs](vision_lora_merge/configs) directory before running the code.

The checkpoints we used for Table 2 (FFT merging) are provided [in this link](https://drive.google.com/drive/folders/1fzHAN3v0qDJuHiD3EkQ76K0sc1cCO_S-). Remember to specify the `model_location` with `/your_model_path/fft_checkpoints` in [config.yaml](vision_fft_merge/config/config.yaml) before running the code.

For FFT merging, the pretrained ViT models are automatically downloaded when running the code. For LoRA merging, please download ViT-B-32, ViT-B-16 and ViT-L-14 from HuggingFace to your local disk before running the code. Remember to specify the local path in `MODEL_DIR` and `CACHE_DIR` in scripts under the [configs](vision_lora_merge/configs) directory before running the code.

```bash
huggingface-cli download openai/clip-vit-base-patch32 --local-dir /your_model_path/clip-vit-base-patch32
huggingface-cli download openai/clip-vit-base-patch16 --local-dir /your_model_path/clip-vit-base-patch16
huggingface-cli download openai/clip-vit-large-patch14 --local-dir /your_model_path/clip-vit-large-patch14
```

### Main Results

## Merging Vision-Language Models

### Environment
Create an environment and install dependencies:
```bash
conda env create -f environment_vlm.yml
conda activate dcmerge_vlm
cd ./vlm_merge
```

## Acknowledgements


## Citation
If you find this repository useful for your work, please consider citing our paper:

```bibtex
@inproceedings{zhang2026dcmerge,
  title={DC-Merge: Improving Model Merging with Directional Consistency},
  author={Han-Chen Zhang and Zi-Hao Zhou and Mao-Lin Luo and Shimin Di and Min-Ling Zhang and Tong Wei},
  booktitle={CVPR},
  year={2026}
}
```
