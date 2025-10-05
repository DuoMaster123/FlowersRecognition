# FlowersRecognition

Project for image classification of flowers using PyTorch and `torchvision.datasets.ImageFolder`.

## Structure

```
Download folder form Kaggle, unzip and name it 'flowers', rememver the position you put it in the disk

FlowersRecognition/
├── data/
│   └── raw/flowers            
├── notebooks/
├── src/
│   └── train.py               # Training script using ImageFolder + ResNet18
├── models/
│   └── checkpoints/
├── outputs/
│   └── logs/plots
├── requirements.txt
└── config.yaml
```

## Quick start

1. Put your unzipped Kaggle `flowers` folder into `E:/flowers` (or update `config.yaml` -> `data.root_dir` to the actual path).
2. (Optional) Create a virtualenv and install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run training:
   ```
   python src/train.py --config config.yaml
   ```

The script will:
- Load images with `torchvision.datasets.ImageFolder`
- Split train/val
- Train a ResNet18 for a few epochs and save the best checkpoint to `models/checkpoints/`