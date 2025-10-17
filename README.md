# MAGE-UNet 🔬

> A Multi-scale Attention-Guided Encoder-Decoder with Large-Kernel Fusion for High-Precision Medical Image Segmentation

## 📝 Project Overview

MAGE-UNet is a specialized deep learning model for high-precision medical image segmentation. It combines multi-scale attention mechanisms and large-kernel fusion technology to provide more accurate segmentation results.

![MAGE-UNet Architecture](./MAGE-UNet.jpg)


## 🚀 Quick Start

### 1. Download Pre-trained ViT Models

We use Google's pre-trained ViT models. You can obtain them from:
- Supported model types: R50-ViT-B_16, ViT-B_16, ViT-L_16, etc.
- [Download models here](https://console.cloud.google.com/storage/vit_models/)

Download and move the model files:
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
mkdir ../model/vit_checkpoint/imagenet21k
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. Prepare Datasets

All required datasets are readily available:
- [BTCV preprocessed dataset](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing)
- [ACDC dataset](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4?usp=drive_link)

### 3. Environment Setup

Requirements:
- Python 3.7
- Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Training and Testing

#### Training
Train the model on the Synapse dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```
> Note: The batch size can be reduced to 12 or 6 to save memory (please decrease the base_lr linearly accordingly). Both settings can achieve similar performance.

#### Testing
Support testing for both 2D images and 3D volumes:
```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## 📊 Performance

### Visualization Results

![Segmentation Performance Comparison](./visualization.jpg)

The above figure demonstrates the comparative performance of different segmentation models. From left to right: input image, ground truth mask, MAGE-UNet segmentation result, Swin-UNet model result, and baseline TransUNet result. The comparison clearly shows that MAGE-UNet achieves significant improvements over the baseline models in terms of edge detail preservation and overall segmentation accuracy.

## 📜 License

*To be added...*

## 👥 Contributors

*To be added...*

## 📮 Contact

*To be added...*
