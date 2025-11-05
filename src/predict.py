import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import argparse
import os

# ================== COLORS & STYLE ===================
from colorama import Fore, Style, init
init(autoreset=True)

def fancy_line(char="─", length=45, color=Fore.LIGHTMAGENTA_EX):
    return color + char * length + Style.RESET_ALL

# ANSI 24-bit True Color cho neon cyan
def neon_cyan(text):
    # #70f5ee → RGB(112, 245, 238)
    return f"\033[38;2;112;245;238m{text}\033[0m"

# ================== MODEL LOADING ===================
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    if "classes" in checkpoint:
        class_names = checkpoint["classes"]
        num_classes = len(class_names)
    else:
        print(Fore.YELLOW + "⚠️  Warning: 'classes' not found in checkpoint. Using default 7 classes." + Style.RESET_ALL)
        class_names = ['bellflower', 'daisy', 'dandelion', 'lotus', 'rose', 'sunflower', 'tulip']
        num_classes = len(class_names)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, class_names

# ================== PREDICTION ===================
def predict_image(model, image_path, class_names, topk=3, threshold=0.6):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    img_t = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = probs.topk(topk, dim=1)

    top_probs = top_probs[0].tolist()
    top_indices = top_indices[0].tolist()

    if top_probs[0] < threshold:
        return [("Cannot predict: Confidence below threshold. "
                 "This image may not belong to the seven flower species in the dataset, "
                 "or it may be unrelated to flowers.", top_probs[0])]
    results = [(class_names[i], top_probs[idx]) for idx, i in enumerate(top_indices)]
    return results

# ================== MAIN FUNCTION ===================
def main():
    parser = argparse.ArgumentParser(description="Flower Prediction")
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--topk', type=int, default=3, help='Show top-K predictions')
    parser.add_argument('--threshold', type=float, default=0.6, help='Confidence threshold (0–1)')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(Fore.RED + f"✖ Image not found: {args.image}")
        return

    if not os.path.exists(args.model):
        print(Fore.RED + f"✖ Model not found: {args.model}")
        return

    print(fancy_line())
    print(Fore.CYAN + "Loading model..." + Style.RESET_ALL)
    model, class_names = load_model(args.model)
    print(Fore.GREEN + f"Model loaded successfully. ({len(class_names)} classes)" + Style.RESET_ALL)
    print(fancy_line())

    print(Fore.CYAN + f"Predicting image: {args.image}" + Style.RESET_ALL)
    results = predict_image(model, args.image, class_names, args.topk, args.threshold)

    print(fancy_line())

    if results[0][0].startswith("Cannot predict"):
        # ========== CASE: Prediction failed ==========
        print(Fore.RED + Style.BRIGHT + "⚠️  Prediction could not be made:" + Style.RESET_ALL)
        print(Fore.YELLOW + "  → Reason: " + Style.RESET_ALL + results[0][0])
        print(Fore.LIGHTBLACK_EX + f"  (Highest confidence: {results[0][1]*100:.2f}%)" + Style.RESET_ALL)
        print(fancy_line(color=Fore.RED))
        print(Fore.LIGHTRED_EX + "Suggestion: Try another image or check if the image really contains one of the 7 flower types." + Style.RESET_ALL)
    else:
        # ========== CASE: Successful prediction ==========
        print(neon_cyan("Top predictions:"))
        for idx, (cls, prob) in enumerate(results):
            if idx == 0:
                color = Fore.RED      # Kết quả đầu tiên đỏ
            else:
                color = Fore.YELLOW   # Hai kết quả sau vàng
            bar = "█" * int(prob * 20)  # visual probability bar
            print(f"  {color}{cls:12s} : {prob*100:6.2f}% |{bar}{Style.RESET_ALL}")
    print(fancy_line())

if __name__ == "__main__":
    main()
