# MAGE-UNet 🔬

> A Multi-scale Attention-Guided Encoder-Decoder with Large-Kernel Fusion for High-Precision Medical Image Segmentation

## 📝 项目简介

MAGE-UNet是一个专门用于高精度医学图像分割的深度学习模型。它结合了多尺度注意力机制和大核融合技术，以提供更准确的分割结果。

## 🚀 快速开始

### 1. 下载预训练的ViT模型

我们使用Google预训练的ViT模型。您可以从以下位置获取模型：
- 支持的模型类型：R50-ViT-B_16, ViT-B_16, ViT-L_16等
- [点击这里下载模型](https://console.cloud.google.com/storage/vit_models/)

下载并移动模型文件：
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
mkdir ../model/vit_checkpoint/imagenet21k
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz
```

### 2. 准备数据集

所有需要的数据集都可以直接获取：
- [BTCV预处理数据集](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd?usp=sharing)
- [ACDC数据集](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4?usp=drive_link)

### 3. 环境配置

环境要求：
- Python 3.7
- 安装依赖：
```bash
pip install -r requirements.txt
```

### 4. 训练与测试

#### 训练模型
在Synapse数据集上训练模型：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```
> 注意：可以根据显存情况将batch size调整为12或6（请相应地线性减少base_lr），两种设置都能达到相似的性能。

#### 测试模型
支持对2D图像和3D体积数据进行测试：
```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## 📊 性能展示

*待补充...*

## 📜 许可证

*待补充...*

## 👥 贡献者

*待补充...*

## 📮 联系方式

*待补充...*
