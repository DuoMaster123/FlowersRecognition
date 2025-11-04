### IMPORTANT: 
- This repository does not include the file `best_model_7classes.pth`, which stores the trained model weights.
- The `.pth` file is generated after training the model and is used to save and later reload the model’s learned parameters for inference or fine-tuning.
- Due to GitHub’s file size limit (100 MB), this file cannot be uploaded to the repository. To obtain it, you need to train the model locally or on a server, which will automatically generate the `.pth` file after training. But if you don't want to spend time on training, you can directly download it through: https://drive.google.com/file/d/1sA3PJOpXdLHXT01vF1-I3SnUt_Qsf_bX/view?usp=sharing
- Please use `usth.edu.vn` mail to access. After downloading, create folder \models\checkpoints\ inside the project and put the file into there.
- Ex: `FlowersRecogniton\models\checkpoints\best_model_7classes.pth`

! Source of the Dataset ! --> https://www.kaggle.com/datasets/nadyana/flowers

*** Note: I ran this program on Window, so I can't sure about MacOS or Linux ***

# FlowersRecognition

Project for image classification of flowers using PyTorch and `torchvision.datasets.ImageFolder`.

## Structure

Download folder form Kaggle, unzip and name it 'flowers', remember the position you put it in the local disk

Clone this Github repository, the folder's structure is as follow:
c:\Users\HP\AppData\Local\Packages\MicrosoftWindows.Client.Core_cw5n1h2txyewy\TempState\ScreenClip\{D4F9CB37-B0A7-4EF9-9BFA-C1C96DA84033}.png

## Quick start

1. Put your unzipped Kaggle `flowers` folder into `E:/flowers` (or update `config.yaml` -> paste your path).
2. Create a virtualenv and install requirements (run in VScode Terminal):
   ```
   pip install -r requirements.txt
   ```
3. Run training, run this command in VScode Terminal (note that it might take a long time depending on your Laptop/PC configuration - usually 2 hours)
   ```
   python src/train.py --config config.yaml  
   ```

The script will:
- Load images with `torchvision.datasets.ImageFolder`
- Split train/val
- Train a ResNet18 for 12 epochs then save the best checkpoint to `models/checkpoints/` and print out reports

4. Predict other inputs:
   ```
   python src/predict.py --image "C:\Users\...\Downloads\example.jpg" --model models\checkpoints\best_model_7classes.pth --threshold 0.7
   ```
- You can change the `C:\....` to the position of the image you use for the prediction (inside your file explorer)
- Threshold = 0.5 → more lenient (accepts more images, but higher risk of incorrect predictions)
- Threshold = 0.7 → more strict (only accepts when the model is more confident) --> You should use this when running the prediction command
