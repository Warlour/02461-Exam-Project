import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import CustomDataset
from time import perf_counter
import argparse
import datetime
from math import ceil
from models import EmotionRecognizer
from functions import *
import os, time
import torchvision, matplotlib
import pandas as pd
import sys
import itertools
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
#from plot_results import plot_results

# Table print
from rich.console import Console
from rich.table import Column, Table

parser = argparse.ArgumentParser(
    prog="Emotion Recognizer Model",
    description="Trains and tests the model",
    epilog="Alfred, Ali and Mathias | January 2024, Introduction to Intelligent Systems (02461) Exam Project"
)
parser.add_argument('-b', '--batch_size',     type=int,   default=64,                         help="Batch size | Default: 64")
parser.add_argument('-l', '--learning_rate',  type=float, default=0.01,                       help="Learning rate | Default: 0.01")
parser.add_argument('-e', '--epochs',         type=int,   default=20,                         help="Number of epochs | Default: 20")
parser.add_argument('-w', '--weight_decay',   type=float, default=0.005,                      help="Optimizer weight decay | Default: 0.005")
parser.add_argument('-g', '--gamma',          type=float, default=0.5,                        help="Gamma | Default: 0.5")
parser.add_argument(      '--min_lr',         type=float, default=0,                          help="Minimum LR, also called eta_min in some schedulers | Default: 0")
parser.add_argument(      '--max_lr',         type=float, default=0,                          help="Maximum LR | Default: 0")
parser.add_argument(      '--last_epoch',     type=float, default=-1,                         help="The index of last epoch | Default: -1")
parser.add_argument('-m', '--momentum',       type=float, default=0.9,                          help="Momentum | Default: 0")
parser.add_argument('-o', '--output_csv',     type=str,   default="data.xlsx",                help="CSV output filename | Default: data.csv")

parser.add_argument('-t', '--scheduler_type', type=str,   default="None",                     help="Scheduler type (Case-sensitive) | Default: None")
parser.add_argument('-c', '--disable_csv',                default=False, action='store_true', help="Disable CSV output | Default: False")

# Testing
parser.add_argument(      '--note',           type=str,   default="",                         help="Note")
parser.add_argument(      '--optimizerfunc',  type=str,   default="",                         help="Optimizer function to be saved in Excel file")
parser.add_argument(      '--loss_function',  type=str,   default="CEL",                      help="Loss function to be saved in Excel file")

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
momentum = args.momentum

'''
Scheduler types:
- None
- AliLR
- CosineAnnealingLR
- StepLR
- MultiStepLR
- MultiplicativeLR
- ExponentialLR
- CyclicLR
- OneCycleLR
- CosineAnnealingWarmRestarts
'''
scheduler_type = args.scheduler_type

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using", device)

if scheduler_type == "None":
    print("Scheduler disabled")
    gamma = 0

# Information for csv output (change manually)
loss_function = args.loss_function
optimizerfunc = args.optimizerfunc
dataset = "FER2013"
convlayers = 4
pools = 2
user = "Alfred"
note = args.note

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
    "Scheduler":            [scheduler_type], 
    "Min. LR":              [min_lr]
}

# Load dataset
"""
all_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5],
                                                          std=[0.5])
                                    ])

train_dataset = CustomDataset(root='data/FER2013/train', transform=all_transforms)

test_dataset = CustomFER2013Dataset(root='data/FER2013/test', transform=all_transforms)
"""
# Augmentation for the training dataset
train_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(10),      # Randomly rotate images by up to 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Randomly change brightness, contrast, and saturation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Only basic transforms for the test dataset
test_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = CustomDataset(root='data/FER2013/train', transform=train_transforms)
test_dataset = CustomDataset(root='data/FER2013/test', transform=test_transforms)



train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

model = EmotionRecognizer(num_classes).to(device)
lossfunction = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

# Use most suitable parameters provided for scheduler
if scheduler_type != "None" and scheduler_type != "AliLR":
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

# Ali custom scheduler
elif scheduler_type == "AliLR":
    milestep = [i for i in range(ceil(num_epochs/10), num_epochs + 1, ceil(num_epochs/10))]

# Variables
total_step = len(train_loader)
times = []
N = 30  # Number of step times to display and consider in the moving average
steptimes = []

# For plotting results
#train_losses = []
#test_losses = []
#accuracies = []
#epochs_list = []

# Training
losses = []
train_losses = []
test_losses = []

#early stoppage
early_stop_counter = 0
patience_threshold = 5  # Number of epochs to wait before stopping
min_test_loss = float('inf')

for epoch in range(num_epochs):
    start = perf_counter()
    steptimes = []

    # Ali custom scheduler per epoch or if above 10 epochs, every 10th epoch
    if scheduler_type == "AliLR":
        if epoch+1 in milestep:
            print("Learning rate decreased for this epoch.")
            if get_lr(optimizer)*gamma > min_lr:
                set_lr(optimizer, get_lr(optimizer)*gamma)
            else:
                set_lr(optimizer, min_lr)

    for i, (images, labels) in enumerate(train_loader):
        stepstart = perf_counter()
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = lossfunction(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Scheduler step
        if scheduler_type != "None" and scheduler_type != "AliLR":
            try:
                scheduler.step()
            except ValueError:
                print("ValueError: Trying to step scheduler beyond the last epoch. Exiting...")
                sys.exit(0)

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

    # training error/loss
    train_losses.append(losses[-1])

    # Testing phase
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_loss += lossfunction(outputs, labels).item()
    
    test_loss_avg = test_loss / len(test_loader)  # Record the testing loss
    test_losses.append(test_loss_avg)

    if test_loss_avg < min_test_loss:
        min_test_loss = test_loss_avg
        early_stop_counter = 0  # reset counter if test loss decreases
    else:
        early_stop_counter += 1  # increment counter if test loss does not decrease

    if early_stop_counter > patience_threshold:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break  # Break out of the training loop
    model.train()

    if scheduler_type != "None" and scheduler_type != "AliLR":
        last_lr = scheduler.get_last_lr()[0]
    elif scheduler_type == "AliLR":
        last_lr = get_lr(optimizer)
    else:
        last_lr = learning_rate

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {losses[-1]} | Test Loss: {test_loss_avg} | Time elapsed: {measure:.2f}s | Learning rate: {last_lr}")

ct = datetime.datetime.now()
new_data["Loss"] = [losses[-1]]
new_data["Min. Loss"] = [min(losses)]

# Testing
print("Testing...", end="\r")
#initializing test_loss
#test_loss = 0
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    #test loss/error
    #test_loss /= len(test_loader.dataset)
    #test_losses.append(test_loss)

    accuracy = correct/total
    #appending accuracy to accuracies lsit for plotting
    #accuracies.append(accuracy)
    new_data["Accuracy"] = [accuracy]
    new_data["Avg. Time / Epoch"] = [round(sum(times)/len(times), 1)]

    #epoch list for plotting
    #epochs_list.append(epoch + 1)

#plot training error vs test error
#plot_results(epoch + 1, train_losses, test_losses, accuracies)

ct_text = f"{ct.year}-{ct.month}-{ct.day} {ct.hour}.{ct.minute}.{ct.second}"

if not os.path.exists(f"models/{note}"):
    os.makedirs(f"models/{note}")
# Save model
customname = f"{ct_text} b{batch_size}-l{last_lr}-e{num_epochs}-w{weight_decay}-g{gamma}-ml{min_lr}-m{momentum} a{accuracy:.1f} {loss_function}-{optimizerfunc}-{scheduler_type}"
torch.save(model.state_dict(), f"models/{note}/{customname}.pt")

# Testing information
ct_text = f"{ct.year}-{ct.month}-{ct.day} {ct.hour}:{ct.minute}:{ct.second}"
new_data['Date & Time'] = [ct_text]
new_data['Total training time'] = [round(sum(times), 1)]
new_data['Learning rate'] = [last_lr]

# Write to Excel
if not args.disable_csv:
    while True:
        try:
            # Input
            try:
                df = pd.read_excel(args.output_csv)
            except FileNotFoundError:
                df = pd.DataFrame(columns=new_data.keys())
                df.to_excel("Excel/"+args.output_csv, index=False)
                print("Created new Excel file")
            
            df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)

            # Output
            df.to_excel(args.output_csv, index=False)
            break
        except PermissionError:
            print("Please close the Excel file. Retrying in 3 seconds...")
            time.sleep(3)
    print("Wrote to Excel")

# Pretty print table
console = Console()
table = Table(show_header=True, header_style="bold magenta")

data = []
for key, value in new_data.items():
    table.add_column(key)
    data.append(str(value[0]))

table.add_row(*data)

console.print(table)

# torchvision.utils.make_grid()
# matplotlib.pyplot.imshow()

import matplotlib.pyplot as plt

# Number of epochs actually completed
actual_epochs = len(test_losses)

# Plotting the training and testing losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, actual_epochs + 1), train_losses[:actual_epochs], label='Training Loss')
plt.plot(range(1, actual_epochs + 1), test_losses, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Testing Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_testing_loss_plot.png')  # Save the plot as a PNG file
plt.show()