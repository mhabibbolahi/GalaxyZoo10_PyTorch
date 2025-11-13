import torch
from torchvision import transforms
from PIL import Image
from cnn import CNNModel
import time
# open('static/b'Abyssinian_1.jpg'.jpg')

def preprocess_image(img_path, device):
    time.sleep(1)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    return input_tensor


def main(img_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = preprocess_image(img_path, device)

    checkpoint = torch.load(model_path, map_location=device)

    state_dict = checkpoint["model_state_dict"]

    model = CNNModel().to(device)
    model.load_state_dict(state_dict)
    model.eval()    

    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()
    class_names = [
        "Disturbed Galaxies",  # 0
        "Merging Galaxies",  # 1
        "Round Smooth Galaxies",  # 2
        "In-between Round Smooth Galaxies",  # 3
        "Cigar Shaped Smooth Galaxies",  # 4
        "Barred Spiral Galaxies",  # 5
        "Unbarred Tight Spiral Galaxies",  # 6
        "Unbarred Loose Spiral Galaxies",  # 7
        "Edge-on Galaxies without Bulge",  # 8
        "Edge-on Galaxies with Bulge"  # 9
    ]
    return class_names[predicted_class]





