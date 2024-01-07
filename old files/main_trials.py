import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from datasets import SubfoldersDataset
from time import perf_counter
import argparse
import datetime
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim.lr_scheduler as lr_scheduler
from math import ceil

parser = argparse.ArgumentParser(
    prog="Emotion Recognizer Model",
    description="Trains and tests the model",
    epilog="Alfred, Ali and Mathias | January 2024, Introduction to Intelligent Systems (02461) Exam Project"
)
parser.add_argument('-b', '--batch_size', type=int, nargs=1, default=[64], help="Batch size | Default: 64")
parser.add_argument('-l', '--learning_rate', type=float, nargs=1, default=[0.001], help="Learning rate | Default: 0.01")
parser.add_argument('-e', '--epochs', type=int, nargs=1, default=[20], help="Number of epochs | Default: 20")
parser.add_argument('-w', '--weight_decay', type=float, nargs=1, default=[0.005], help="Weight Decay | Default: 0.005")
parser.add_argument('-c', '--enable_csv', default=True, action='store_false', help="Toggle CSV output | Default: True")
parser.add_argument('-lrd', '--learning_rate_decay', type=str, nargs=1, default=["None"], help="learning rate decay | Default: None")
parser.add_argument('-o', '--output_csv', type=str, nargs=1, default=["data.xlsx"], help="CSV output filename | Default: data.csv")
args = parser.parse_args()

# Subset of training dataset that is processed together during a single iteration of the training algorithm
batch_size = args.batch_size[0]
# Number of feelings
num_classes = 7
learning_rate = args.learning_rate[0]
num_epochs = args.epochs[0]
weight_decay = args.weight_decay[0]
learning_rate_decay = args.learning_rate_decay[0]


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using", device)

if args.enable_csv:
    import pandas as pd
    pd.set_option('max_colwidth', 17)

    # Information for csv output (change manually)
    loss_function = "CEL"
    optimizerfunc = "Adam"
    dataset = "FER2013"
    convlayers = 4
    pools = 2
    user = "Alfred"

    new_data = {"Date & Time": [], "Epochs": [num_epochs], "Batch size": [batch_size], "Learning rate": [learning_rate], "Optimizer function": [optimizerfunc], "Loss function": [loss_function], "Avg. Time / Epoch": [], "Image dimension": [32], "Loss": [], "Min. Loss": [], "Accuracy": [], "Dataset": [dataset], "Device": [device], "Convolutional layers": [convlayers], "Pools": [pools], "Created by": [user], "Weight decay": [weight_decay], "Learning rate decay": [learning_rate_decay]}

    

all_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5],
                                                          std=[0.5])
                                    ])

train_dataset = SubfoldersDataset(root='data/FER2013/train', transform=all_transforms)

test_dataset = SubfoldersDataset(root='data/FER2013/test', transform=all_transforms)

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
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        '''Fully connected layers tilknytter hver neuron til næste neuron'''
        self.fc1 = nn.Linear(256, 128) # Fully connected layer
        self.relu1 = nn.ReLU() # Aktiveringsfunktion
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.max_pool1(out)
        out = self.conv_layer2(out)

        out = self.conv_layer3(out)
        out = self.max_pool2(out)
        out = self.conv_layer4(out)
        out = self.max_pool3(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        out=self.fc2(out)
        return out
#if torch.cuda.is_available():
#    model = EmotionRecognizer(num_classes).to(device).half()
#    model = model.to(dtype=torch.float16)
#else:
#    model = EmotionRecognizer(num_classes).to(device)

model = EmotionRecognizer(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)
total_step = len(train_loader)

#total_steps = len(train_loader) * num_epochs  # Corrected total_steps
#scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=total_steps)  # OneCycle
#milestones = [int(i * num_epochs / 5) for i in range(1, 5)]
#scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5) #MultiStep
#scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs) #CosineAnnealing
#scheduler = ExponentialLR(optimizer, gamma=0.95, last_epoch=-1) #Exponetial
#scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # StepLR

times = []

N = 30  # Number of step times to display and consider in the moving average
steptimes = []

# Training
losses = []
for epoch in range(num_epochs):
    start = perf_counter()
    steptimes = []
    for i, (images, labels) in enumerate(train_loader):
        stepstart = perf_counter()
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

        stepend = perf_counter()
        steptimes.append(stepend - stepstart)

        # Calculate ETA every N steps
        if i % N == 0 and i > 0:
            avg_steptime = sum(steptimes) / len(steptimes)
            remaining_steps = total_step - i
            eta = avg_steptime * remaining_steps

            # Calculate drift factor
            actual_time = stepend - start
            estimated_time = i * avg_steptime
            drift_factor = actual_time / estimated_time if estimated_time > 0 else 1

            # Adjust ETA by drift factor
            eta *= drift_factor

            print(f" {i}/{total_step} | ETA: {eta:.0f}s           ", end="\r")

    end = perf_counter()

    measure = end - start
    times.append(measure)
    losses.append(round(loss.item(), 4))
    print(f'Epoch [{epoch+1}/{num_epochs}] | Loss: {losses[-1]} | Time elapsed: {measure:.2f}s')

ct = datetime.datetime.now()
if args.enable_csv:
    new_data["Loss"] = [losses[-1]]
    new_data["Min. Loss"] = [min(losses)]

# Testing
print("Testing...", end="\r")
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
    
    accuracy = round(100*correct/total, 4)
    new_data["Accuracy"] = [accuracy]
    print(f"{accuracy} % Accurate | Trained on {total_step*batch_size} images")
    avgtimeepoch = round(sum(times)/len(times), 1)
    new_data["Avg. Time / Epoch"] = [avgtimeepoch]
    print(f"Average epoch time: {avgtimeepoch} seconds")
    print(f"Batch size: {batch_size} | Learning rate: {learning_rate}")

ct_text = f"{ct.year}-{ct.month}-{ct.day} {ct.hour}.{ct.minute}.{ct.second}"

# Save model
#best_model_state = copy.deepcopy()
torch.save(model.state_dict(), f"models/{ct_text} b{batch_size}-e{num_epochs}-a{accuracy} {loss_function}-{optimizerfunc}.pt")

if args.enable_csv:
    ct_text = f"{ct.year}-{ct.month}-{ct.day} {ct.hour}:{ct.minute}:{ct.second}"
    new_data['Date & Time'] = [ct_text]
    new_data['Total training time'] = [sum(times)]

    # CSV Format in list
    # Date & Time   Epochs   Batch size   Learning rate   Optimizer function   Loss function   Avg. Time/Epoch   Image dimension   Loss   Min. Loss   Accuracy   Dataset   Device   Convolutional layers   Pools   Created by   Total training time
    #      0          1         2             3                   4                 5                6                 7            8         9          10        11        12            13               14        15                 16
    while True:
        try:
            # Input
            try:
                df = pd.read_excel(args.output_csv[0])
            except FileNotFoundError:
                df = pd.DataFrame(columns=new_data.keys())
                df.to_excel(args.output_csv[0], index=False)
                print("Created new Excel file")
            
            df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)

            # Output
            df.to_excel(args.output_csv[0], index=False)
            break
        except PermissionError:
            input("Please close the Excel file. Press Enter to continue...")
    print("Wrote to Excel")
