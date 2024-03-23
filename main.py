import os
import torchvision
from PIL import Image

import torch
import torch.nn as nn

torch.cuda.empty_cache()

torch.set_default_device("cuda")

images = torch.tensor([], dtype=torch.float32)
classes = []

test_images = torch.tensor([], dtype=torch.float32)
test_classes = []

names = {"Non_Demented": 0, "Very_Mild_Demented": 1, "Mild_Demented": 2, "Moderate_Demented": 3}

for folder in sorted(os.listdir("Dataset")):
    for idx, file in enumerate(sorted(os.listdir(f"Dataset/{folder}"))):
        if idx > len(os.listdir(f"Dataset/{folder}")) * 0.2: continue
        img = Image.open(f"Dataset/{folder}/{file}").resize((128, 128)).convert("L")
        torch_image = torchvision.transforms.ToTensor()(img).to("cuda")

        if idx < len(os.listdir(f"Dataset/{folder}")) * 0.15:
            images = torch.cat((images, torch_image.unsqueeze(0)), dim=0)
            temp = [0, 0, 0, 0]
            temp[names[folder]] = 1
            classes.append(temp)
        else:
            test_images = torch.cat((test_images, torch_image.unsqueeze(0)), dim=0)
            temp = [0, 0, 0, 0]
            temp[names[folder]] = 1
            test_classes.append(temp)

y = torch.tensor(classes, dtype=torch.float32)

print(y.size())

print(images.size())


class AlzheimerClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Grayscale images have 1 channel
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # Grayscale images have 1 channel
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # Grayscale images have 1 channel
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 2048),  # Adjust based on output of conv layers
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


model = AlzheimerClassifier()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 500
model.train()
for epoch in range(epochs):
    outputs = model(images)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss}')

torch.save(model.state_dict(), 'model.pt')

model.eval()
with torch.inference_mode():
    pred = torch.nn.functional.softmax(model(test_images), dim=1)
    print(pred)
    accurate = 0
    for idx, p in enumerate(pred):
        if int(torch.max(p, 0).indices.item()) == test_classes[idx].index(max(test_classes[idx])):
            accurate += 1

    print(accurate, len(pred), accurate / len(pred))
