import os
import torchvision
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

torch.cuda.empty_cache()

torch.set_default_device("cuda")


names = {"NonDemented": 0, "VeryMildDemented": 1, "MildDemented": 2, "ModerateDemented": 3}

transform = transforms.Compose([
    transforms.Grayscale(),  # Convert image to grayscale
    transforms.Resize((128, 128)),  # Resize image
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize pixel values
])

class AlzheimerClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, padding=1),  # Grayscale images have 1 channel
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),  # Grayscale images have 1 channel
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),  # Grayscale images have 1 channel
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12288, 1024),  # Adjust based on output of conv layers
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class ImageFolderDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.classes = []
        self.names = {"NonDemented": 0, "VeryMildDemented": 1, "MildDemented": 2, "ModerateDemented": 3}
        self.train = train

        for folder in sorted(os.listdir(data_dir)):
            for idx, file in enumerate(sorted(os.listdir(f"{data_dir}/{folder}"))):
                split = len(os.listdir(f"{data_dir}/{folder}")) * 0.8
                if (self.train and idx > split) or (not self.train and idx < split): continue
                img_path = f"{data_dir}/{folder}/{file}"
                label = [0.0] * 4  # One-hot encoded label (all zeros)
                label[self.names[folder]] = 1.0  # Set 1 for the corresponding class

                self.images.append(img_path)
                self.classes.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.classes[idx]

        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# Create the dataset and data loader
train_data = ImageFolderDataset("Dataset", transform=transform, train=True)
test_data = ImageFolderDataset("Dataset", transform=transform, train=False)  # Use the same transform
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, generator=torch.Generator(device='cuda'))
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, generator=torch.Generator(device='cuda'))


model = AlzheimerClassifier()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 250
model.train()


for epoch in range(epochs):
    for images, labels in train_loader:
        # Move data to GPU if available
        images = images.to("cuda")
        labels = labels.to("cuda")

        # Forward pass, calculate loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on test set (optional)
    with torch.inference_mode():
        model.eval()  # Set model to evaluation mode
        test_loss = 0
        correct = 0
        for idx, (images, labels) in enumerate(test_loader):
            images = images.to("cuda")
            labels = labels.to("cuda")

            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            # print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            for pred, l in zip(predicted, labels):
                correct += 1 if pred == torch.max(l, 0).indices else 0

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)

        print(f"Epoch: {epoch + 1}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")
