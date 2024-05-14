from collections import OrderedDict
import os

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16384, 4)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.lin(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


image_size = (128, 128)

# Define data transformations
transform = torchvision.transforms.Compose([
    # transforms.Grayscale(),  # Convert image to grayscale
    torchvision.transforms.Resize(image_size),  # Resize image
    torchvision.transforms.ToTensor(),  # Convert image to PyTorch tensor
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize pixel values
])


class ImageFolderDataset(Dataset):
    #dataloader
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.classes = []
        self.names = {folder: idx for idx, folder in enumerate(sorted(os.listdir(data_dir)))}
        self.train = train

        for folder in sorted(os.listdir(data_dir)):
            for idx, file in enumerate(sorted(os.listdir(f"{data_dir}/{folder}"))):
                if idx > len(os.listdir(f"{data_dir}/{folder}")) * 0.8 and self.train \
                        or idx < len(os.listdir(f"{data_dir}/{folder}")) * 0.8 and not self.train:
                    continue
                img_path = f"{data_dir}/{folder}/{file}"
                label = [0.0] * 4  # One-hot encoded label (all zeros)
                label[self.names[folder]] = 1.0  # Set 1 for the corresponding class

                self.images.append(img_path)
                self.classes.append(label)

                if idx % 100 == 0:
                    print(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.classes[idx]

        img = Image.open(img_path).convert("RGB")  #convert to rgb bc unet
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)


train_data = ImageFolderDataset("Dataset", transform=transform)
test_data = ImageFolderDataset("Dataset", transform=transform, train=False)  # Use the same transform
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, generator=torch.Generator(device='cuda'))
test_loader = DataLoader(test_data, batch_size=64, shuffle=False,
                         generator=torch.Generator(device='cuda'))  # Don't shuffle test data

torch.set_default_device("cuda")

model = UNet()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 500
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

        print(
            f"Epoch: {epoch + 1}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")

print("Model training complete!")