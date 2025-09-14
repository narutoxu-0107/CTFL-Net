# CTFL-Net: A Wavelet-Enhanced CNN-Transformer Model for Rapid, Low-Cost Rice LAI Estimation from UAV-RGB Images

## Introduction

This repository is the official PyTorch implementation of the paper "CTFL-Net: A Wavelet-Enhanced CNN-Transformer Model for Rapid, Low-Cost Rice LAI Estimation from UAV-RGB Images".

CTFL-Net is an innovative deep learning model designed for estimating rice Leaf Area Index (LAI) using only affordable UAV-RGB images. The core idea is to combine local texture features with global contextual information via a dual-branch network architecture:

* **Local Branch:** Utilizes ConvNeXt-Tinyas the backbone to effectively extract fine-grained local features.
* **Global Branch:** Employs DaViT-Tiny (Dual Attention Vision Transformer)as the backbone to capture global dependencies.

## Preparation

### RLU Dataset

This project uses our self-constructed **Rice LAI UAV-RGB (RLU) dataset**, which serves as a valuable resource for rice LAI estimation research.

* **Data Collection**:
  * **Location**: Nanchang, Jiangxi Province, China (N:28°31'10'', E:116°4'6'').
  * **Equipment**: A DJI Phantom 4 Advanced quadcopter UAV equipped with a consumer-grade RGB camera.
  * **Method**: The UAV flew at a low altitude of 7 meters, capturing high-resolution images from a vertical downward perspective using a hovering shooting method to ensure image quality.
  * **Time**: Data was collected from 2018 to 2020 during two key growth stages of rice, typically every three days.

* **Ground Truth**:
  * The ground truth LAI was measured using the high-precision **specific leaf weight method**, which involves destructive sampling to ensure label accuracy

* **Dataset Scale and Processing**:
  * The dataset comprises a total of **9,500 images**.
  * Original images with a resolution of 5472×3648 were cropped into 1024×1024 patches for model training and testing.

#### Download Links

You can download the datasets and pre-trained model weights from the following links:

**Datasets:**
- [**Original RGB Data**](https://pan.baidu.com/s/1H3JTJPKfWTzMC-ukeyAtMw?pwd=sgqf ) - Raw UAV-RGB images (5472×3648 resolution)
- [**RLU Dataset (Processed)**](https://pan.baidu.com/s/179GHv49sMfLGgccjHPy82w?pwd=vem3 ) - Processed dataset ready for model training (1024×1024 patches)

**Pre-trained Model Weights:**
- [**CTFL-Net Trained Weights**](https://pan.baidu.com/s/19-vjj8ZrkG9e45ECLTbsgQ?pwd=vh6x ) - Pre-trained CTFL-Net model weights for rice LAI estimation

After downloading, please organize the files according to the directory structure described below.

### 1. Download Pre-trained Weights

This model uses `ConvNeXt-Tiny` and `DaViT-Tiny` from the `timm` library as pre-trained backbones. To load the model successfully, please download their pre-trained weight files and place them in the `ckpt` folder in the project's root directory.

```
CTFL-Net/
├── timm/
│   ├── ckpt/
│       ├── image1.jpg timm-convnext_tiny.in12k_ft_in1k.safetensors
│       └── image1.jpg timm-davit_tiny.msft_in1k.bin
├── train_scripts/
└── ...
```

* **ConvNeXt-Tiny**: [Download Link](https://pan.baidu.com/s/10XSfpbS7qd1Xegfh9t9wuw?pwd=p6ft) (Please rename the file as specified in the code after downloading)
* **DaViT-Tiny**: [Download Link](https://pan.baidu.com/s/1psbj2eAERL0wD-CHO9OT-w?pwd=tepe) (Please rename the file as specified in the code after downloading)

*Note: Please adjust filenames based on `timm/models/ctflnet.py`.*

### 2. Dataset Directory Structure

Please organize your training and validation datasets according to the following structure:

```
/path/to/your/dataset/RLU
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image1.jpg
│   └── class2/
│       ├── image3.jpg
│       └── ...
├── 2019-1/
│   └── ...
├── 2019-2/
│   └── ...
└── 2020-2/
    └── ...
```

Then, modify the data paths in the `train_scripts/ctflnet.yaml` configuration file:

```yaml
data_dir: /path/to/your/dataset/URL/train/  # Path to the training set
data_val_dir_0: /path/to/your/dataset/URL/2019-1 # Path to validation set 1
data_val_dir_1: /path/to/your/dataset/URL/2019-2 # Path to validation set 2
data_val_dir_2: /path/to/your/dataset/URL/2020-1 # Path to validation set 3
```

*Note: Based on your `train.py` script, the training set appears to be hard-coded to use data from 2018. Please ensure your data loading logic matches this configuration.*

## Training

You can train the CTFL-Net model by running the following command. All training parameters can be configured in `train_scripts/ctflnet.yaml`.

```bash
python train.py --config train_scripts/ctflnet.yaml
```

Weight files and logs will be saved in the `output/train/` directory.

## Evaluation

We provide a `validate.py` script for evaluating model performance, which supports flexible configuration via command-line arguments.

A basic evaluation command is as follows:

```bash
python validate.py \
    --model ctflnet \
    --checkpoint /path/to/your/best_model.pth \
    --data-dir /path/to/your/validation/dataset \
```

The script will output real-time metrics for each batch and a final summary for the entire validation set, including RMSE, R², rRMSE, MAE, and nRMSE.

## Citation

If you use the code or dataset from this repository in your research, please consider citing our paper:

```bibtex
@article{Bai2024Estimation,
  title={Estimation of Regional Rice Leaf Area Index via Low-altitude UAV and Affordable RGB Camera Remote Sensing},
  author={Bai, Xiaodong and Xu, Jiajing and Liu, Xiaozhang and Chen, Da and Yang, Aiping and Wang, Jianjun},
  journal={[Journal or Conference Name]},
  year={2024},
  publisher={[Publisher]}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgements

The implementation of this project heavily relies on the [PyTorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models) library. We sincerely thank the original author, Ross Wightman, for his excellent work.