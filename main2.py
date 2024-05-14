import torch
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image


# Define paths and hyperparameters
data_dir = "Dataset"
image_size = (128, 128)  # Size of images
num_classes = 4  # Number of classes in dataset
batch_size = 32  # Batch size for training
learning_rate = 0.001  # Learning rate for optimizer
num_epochs = 10  # Number of training epochs

# Define data transformations
transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize image
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize pixel values
])


class ImageFolderDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.classes = []
        self.names = {folder: idx for idx, folder in enumerate(sorted(os.listdir(data_dir)))}

        for folder in sorted(os.listdir(data_dir)):
            for idx, file in enumerate(sorted(os.listdir(f"{data_dir}/{folder}"))):
                if idx > len(os.listdir(f"{data_dir}/{folder}")) * 0.2:
                    continue
                img_path = f"{data_dir}/{folder}/{file}"
                label = [0] * 4  # One-hot encoded label (all zeros)
                label[self.names[folder]] = 1  # Set 1 for the corresponding class

                self.images.append(img_path)
                self.classes.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.classes[idx]

        img = Image.open(img_path).resize((128, 128)).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)


# Load the ResNet18 model without pre-training
model = models.resnet18(pretrained=False)

# Modify the final layer for your 4-class classification
num_ftrs = model.fc.in_features  # Number of features from the last layer
model.fc = torch.nn.Linear(num_ftrs, num_classes)  # Replace final layer

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Create the dataset and data loader
train_data = ImageFolderDataset(data_dir, transform=transform)
test_data = ImageFolderDataset(data_dir, transform=transform)  # Use the same transform
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)  # Don't shuffle test data

# Train the model
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
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
    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        test_loss = 0
        correct = 0
        for images, labels in test_loader:
            images = images.to("cuda")
            labels = labels.to("cuda")

            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)

        print(
            f"Epoch: {epoch + 1}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")

    print("Model training complete!")
