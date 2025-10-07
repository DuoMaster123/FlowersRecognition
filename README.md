<<<<<<< HEAD
**! Source of the Dataset !**
https://www.kaggle.com/datasets/nadyana/flowers/data
=======
! Source of the Dataset ! --> https://www.kaggle.com/datasets/nadyana/flowers
>>>>>>> 8ccf07e (Updating the code, including more features after training)

# FlowersRecognition

Project for image classification of flowers using PyTorch and `torchvision.datasets.ImageFolder`.

## Structure

```
Download folder form Kaggle, unzip and name it 'flowers', remember the position you put it in the disk
<<<<<<< HEAD
=======

Clone this Github repository, the folder's structure is as follow:
>>>>>>> 8ccf07e (Updating the code, including more features after training)

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

<<<<<<< HEAD
1. Put your unzipped Kaggle `flowers` folder into `E:/flowers` (**or update `config.yaml` -> `data.root_dir` to the actual Path min the disk**).
=======
1. Put your unzipped Kaggle `flowers` folder into `E:/flowers` (or update `config.yaml` -> paste your path).
>>>>>>> 8ccf07e (Updating the code, including more features after training)
2. (Optional) Create a virtualenv and install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run training: Open the folder in VScode and run this command in Terminal
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
<<<<<<< HEAD
- Train a ResNet18 for a few epochs and save the best checkpoint to `models/checkpoints/`
=======
- Train a ResNet18 for 12 epochs then save the best checkpoint to `models/checkpoints/` and print out reports
>>>>>>> 8ccf07e (Updating the code, including more features after training)
