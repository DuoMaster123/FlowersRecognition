import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import argparse
import os

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Lấy danh sách lớp từ checkpoint (an toàn, không cần sửa tay)
    if "classes" in checkpoint:
        class_names = checkpoint["classes"]
        num_classes = len(class_names)
    else:
        print("⚠️ Warning: 'classes' not found in checkpoint. Using default 7 classes.")
        class_names = ['bellflower', 'daisy', 'dandelion', 'lotus', 'rose', 'sunflower', 'tulip']
        num_classes = len(class_names)

    # Tạo lại model đúng cấu trúc
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load trọng số
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model, class_names


def predict_image(model, image_path, class_names, topk=3):
    # Transform giống lúc train
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    img_t = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_indices = probs.topk(topk, dim=1)

    top_probs = top_probs[0].tolist()
    top_indices = top_indices[0].tolist()
    results = [(class_names[i], top_probs[idx]) for idx, i in enumerate(top_indices)]
    return results


def main():
    parser = argparse.ArgumentParser(description="Flower Prediction")
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--topk', type=int, default=3, help='Show top-K predictions')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return

    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return

    print("🔹 Loading model...")
    model, class_names = load_model(args.model)

    print(f"🔹 Predicting image: {args.image}")
    results = predict_image(model, args.image, class_names, args.topk)

    print("\n✅ Prediction results:")
    for cls, prob in results:
        print(f"  {cls:12s} - {prob*100:.2f}%")


if __name__ == "__main__":
    main()
