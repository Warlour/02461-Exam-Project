import torch
import torch.nn as nn
import torchvision.transforms as transforms
from customdataset import CustomFER2013Dataset
from time import perf_counter
import argparse
import datetime
from math import ceil
from models import EmotionRecognizer
from functions import *
import os

# Table print
from rich.console import Console
from rich.table import Column, Table

parser = argparse.ArgumentParser(
    prog="Emotion Recognizer Model",
    description="Trains and tests the model",
    epilog="Alfred, Ali and Mathias | January 2024, Introduction to Intelligent Systems (02461) Exam Project"
)
parser.add_argument('-b', '--batch_size',    type=int,   default=64,                         help="Batch size | Default: 64")
parser.add_argument('-l', '--learning_rate', type=float, default=0.01,                       help="Learning rate | Default: 0.01")
parser.add_argument('-e', '--epochs',        type=int,   default=20,                         help="Number of epochs | Default: 20")
parser.add_argument('-wd', '--weight_decay', type=float, default=0.005,                      help="Optimizer weight decay | Default: 0.005")
parser.add_argument('-g', '--gamma',         type=float, default=0.1,                        help="Gamma | Default: 0.1")
parser.add_argument('-ml', '--min_lr',       type=float, default=0,                          help="Minimum LR | Default: 0")
parser.add_argument('-o', '--output_csv',    type=str,   default="data.xlsx",                help="CSV output filename | Default: data.csv")
parser.add_argument('-c', '--disable_csv',               default=False, action='store_true', help="Disable CSV output | Default: False")
parser.add_argument('-ds', '--disable_scheduler',        default=False, action='store_true', help="Disable scheduler| Default: False")

args = parser.parse_args()

# Subset of training dataset that is processed together during a single iteration of the training algorithm
batch_size = args.batch_size
# Number of feelings
num_classes = 7
learning_rate = args.learning_rate
num_epochs = args.epochs
gamma = args.gamma
weight_decay = args.weight_decay
min_lr = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using", device)

if args.disable_scheduler:
    print("Schedular disabled")
    gamma = 0

new_data = {}

if not args.disable_csv:
    import pandas as pd

    # Information for csv output (change manually)
    loss_function = "CEL"
    optimizerfunc = "SGD"
    schedulername = "None"
    dataset = "FER2013"
    convlayers = 4
    pools = 2
    user = "Stationær"
    note = "Test 1"

    new_data = {
        "Date & Time":          [], 
        "Epochs":               [num_epochs], 
        "Batch size":           [batch_size], 
        "Learning rate":        [], 
        "Optimizer function":   [optimizerfunc], 
        "Loss function":        [loss_function], 
        "Avg. Time / Epoch":    [], 
        "Image dimension":      [32], 
        "Loss":                 [], 
        "Min. Loss":            [], 
        "Accuracy":             [], 
        "Dataset":              [dataset], 
        "Device":               [device], 
        "Convolutional layers": [convlayers], 
        "Pools":                [pools], 
        "Created by":           [user],
        "Total training time":  [],
        "Gamma":                [gamma], 
        "Weight decay":         [weight_decay], 
        "Scheduler":            [schedulername], 
        "Min. LR":              [min_lr]
    }


# Load dataset
all_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5],
                                                          std=[0.5])
                                    ])

train_dataset = CustomFER2013Dataset(root='data/FER2013/train', transform=all_transforms)

test_dataset = CustomFER2013Dataset(root='data/FER2013/test', transform=all_transforms)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

model = EmotionRecognizer(num_classes).to(device)
lossfunction = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
if not args.disable_scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

total_step = len(train_loader)

milestep = [i for i in range(ceil(num_epochs/10), num_epochs + 1, ceil(num_epochs/10))]

times = []

N = 30  # Number of step times to display and consider in the moving average
steptimes = []

# Training
losses = []
for epoch in range(num_epochs):
    # if epoch > 0 and epoch+1 in milestep and not args.disable_scheduler:
    #     print("Learning rate decreased")
    #     set_lr(optimizer, get_lr(optimizer) * gamma)
    start = perf_counter()
    steptimes = []
    for i, (images, labels) in enumerate(train_loader):
        stepstart = perf_counter()
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = lossfunction(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not args.disable_scheduler:
            scheduler.step()

        stepend = perf_counter()
        steptimes.append(stepend - stepstart)

        # Calculate ETA every N steps
        if i % N == 0 and i > 0:
            avg_steptime = sum(steptimes) / len(steptimes)
            remaining_steps_current_epoch = total_step - i
            remaining_steps_total = remaining_steps_current_epoch + total_step * (num_epochs - epoch - 1)
            eta = avg_steptime * remaining_steps_current_epoch

            # Calculate drift factor
            actual_time = stepend - start
            estimated_time = i * avg_steptime
            drift_factor = actual_time / estimated_time if estimated_time > 0 else 1

            # Adjust ETA by drift factor
            eta *= drift_factor
            total_eta = avg_steptime * remaining_steps_total * drift_factor

            print(f" {i}/{total_step} | ETA: {eta:.0f}s | Total ETA: {total_eta:.0f}s         ", end="\r")
        

    end = perf_counter()

    measure = end - start
    times.append(measure)
    losses.append(round(loss.item(), 4))
    last_lr = scheduler.get_last_lr()[0] if not args.disable_scheduler else learning_rate
    print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {losses[-1]} | Time elapsed: {measure:.2f}s | Learning rate: {last_lr}")

ct = datetime.datetime.now()
if not args.disable_csv:
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
    
    accuracy = correct/total
    new_data["Accuracy"] = [accuracy]
    avgtimeepoch = round(sum(times)/len(times), 1)
    new_data["Avg. Time / Epoch"] = [avgtimeepoch]

ct_text = f"{ct.year}-{ct.month}-{ct.day} {ct.hour}.{ct.minute}.{ct.second}"

if not os.path.exists(f"models/{note}"):
    os.makedirs(f"models/{note}")
# Save model
torch.save(model.state_dict(), f"models/{note}/{ct_text} b{batch_size}-e{num_epochs}-a{accuracy} {loss_function}-{optimizerfunc}-{schedulername} {note}.pt")

if not args.disable_csv:
    ct_text = f"{ct.year}-{ct.month}-{ct.day} {ct.hour}:{ct.minute}:{ct.second}"
    new_data['Date & Time'] = [ct_text]
    new_data['Total training time'] = [round(sum(times), 1)]
    new_data['Learning rate'] = [last_lr]

    while True:
        try:
            # Input
            try:
                df = pd.read_excel(args.output_csv)
            except FileNotFoundError:
                df = pd.DataFrame(columns=new_data.keys())
                df.to_excel(args.output_csv, index=False)
                print("Created new Excel file")
            
            df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)

            # Output
            df.to_excel(args.output_csv, index=False)
            break
        except PermissionError:
            input("Please close the Excel file. Press Enter to continue...")
    print("Wrote to Excel")

console = Console()
table = Table(show_header=True, header_style="bold magenta")

data = []
for key, value in new_data.items():
    table.add_column(key)
    data.append(str(value[0]))

table.add_row(*data)

console.print(table)