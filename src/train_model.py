import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Path
train_dir = "data/dataset/train"

# Image transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

# Dataset load
dataset = datasets.ImageFolder(train_dir, transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print("Classes:", dataset.classes)

# Simple CNN model
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

model = SimpleCNN()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 5

for epoch in range(epochs):
    total_loss = 0
    for imgs, labels in loader:
        preds = model(imgs)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model.pth")

print("✅ Model saved at models/model.pth")
