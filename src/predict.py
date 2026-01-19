import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn

# Same model structure as training
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Load model
model = SimpleCNN()
model.load_state_dict(torch.load("models/model.pth"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    return "FAKE" if pred == 0 else "REAL"

# Test run
if __name__ == "__main__":
    test_image = "data/dataset/train/fake/frame_3.jpg"
 # change if needed
    result = predict_image(test_image)
    print("Prediction:", result)
