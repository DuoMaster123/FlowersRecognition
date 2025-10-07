! Source of the Dataset ! --> https://www.kaggle.com/datasets/nadyana/flowers

*** Note: I ran this program on Window, so I can't sure about MacOS or Linux ***

# FlowersRecognition

Project for image classification of flowers using PyTorch and `torchvision.datasets.ImageFolder`.

## Structure

```
Download folder form Kaggle, unzip and name it 'flowers', remember the position you put it in the local disk

Clone this Github repository, the folder's structure is as follow:

FlowersRecognition/
├── data/
│   └── raw/flowers            # You can link the the folder 'flowers' using File Explorer's path 
├── notebooks/
├── outputs/
│   └──plots/
│   └──reports/
├── src/
│   └── train.py               # Training script using ImageFolder + ResNet18
├── models/
│   └── checkpoints/           # Note that after training completed, there will be some more files to be created
├── outputs/
│   └── logs/plots
├── requirements.txt
└── config.yaml
```

## Quick start

1. Put your unzipped Kaggle `flowers` folder into `E:/flowers` (or update `config.yaml` -> paste your path).
2. (Optional) Create a virtualenv and install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run training: Open the folder in VScode and run this command in Terminal
   ```
   python src/train.py --config config.yaml  
   ```

The script will:
- Load images with `torchvision.datasets.ImageFolder`
- Split train/val
- Train a ResNet18 for 12 epochs then save the best checkpoint to `models/checkpoints/` and print out reports

4. Predict other inputs:
   ```
   python src/predict.py --image "C:\Users\HP\Downloads\example.jpg" --model models\checkpoints\best_model_7classes.pth --threshold 0.7
   ```
- You can change the `C:\....` to the position of the image you use for the prediction (in your folder)
- Threshold = 0.5 → more lenient (accepts more images, but higher risk of incorrect predictions)
- Threshold = 0.7 → more strict (only accepts when the model is more confident) --> You should use this when running the prediction command
