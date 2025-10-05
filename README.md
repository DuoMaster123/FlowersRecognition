**! Source of the Dataset !**
https://www.kaggle.com/datasets/nadyana/flowers/data

# FlowersRecognition

Project for image classification of flowers using PyTorch and `torchvision.datasets.ImageFolder`.

## Structure

```
Download folder form Kaggle, unzip and name it 'flowers', remember the position you put it in the disk

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

1. Put your unzipped Kaggle `flowers` folder into `E:/flowers` (**or update `config.yaml` -> `data.root_dir` to the actual Path min the disk**).
2. (Optional) Create a virtualenv and install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run training:
   ```
   python src/train.py --config config.yaml
   ```
4. Predict the input:
   ```
   python src\predict.py --image "C:\Users\HP\Downloads\example.jpg" --model models\checkpoints\best_model_7classes.pth --topk 3  #You can change the Path to the position the image is saved in disk
   ```

The script will:
- Load images with `torchvision.datasets.ImageFolder`
- Split train/val
- Train a ResNet18 for a few epochs and save the best checkpoint to `models/checkpoints/`
