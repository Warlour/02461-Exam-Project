# https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Subset of training dataset that is processed together during a single iteration of the training algorithm
batch_size = 64
# Number of feelings
num_classes = 7
learning_rate = 0.001
num_epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5],
                                                          std=[0.5])
                                    ])


train_dataset = torchvision.datasets.FER2013(root = './data',
                                             transform = all_transforms)

test_dataset = torchvision.datasets.FER2013(root = './data',
                                             split = "test",
                                             transform = all_transforms)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

class EmotionRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(EmotionRecognizer, self).__init__()
        # in_channels: color channels, black and white is 1, rgb is 3
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        '''Fully connected layers tilknytter hver neuron til næste neuron'''
        self.fc1 = nn.Linear(1600, 128) # Fully connected layer
        self.relu1 = nn.ReLu() # Aktiveringsfunktion
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        out=self.fc2(out)
        return out


model = EmotionRecognizer(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)
total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f"Accuracy of the network on the {28709} train images: {100*correct/total}%")