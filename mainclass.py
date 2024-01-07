import torch
import torch.nn as nn
import torchvision.transforms as transforms

import math
from time import perf_counter
import datetime
import pandas as pd
import time
import matplotlib.pyplot as plt

# Repeat training
import multiprocessing, os, subprocess

from models import *
from datasets import *


class ModelHandler:
    def __init__(self, model, batch_size: int, start_lr: float, epochs: int, gamma: float, weight_decay: float, min_lr: float, momentum: float,
                 datapath: str = "FER2013") -> None:
        # Variables
        self.batch_size = batch_size
        self.classes = 7
        self.start_lr = start_lr
        self.epochs = epochs
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.momentum = momentum

        # To be changed data
        self.accuracy = -1

        '''MODEL'''
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model(self.classes).to(self.device)

        self.data = {
            "Date & Time": [], #
            "Epochs": [self.epochs], #
            "Batch size": [self.batch_size], #
            "Start LR": [self.start_lr], #
            "End LR": [], #
            "Loss function": [], #
            "Optimizer": [], #
            "Scheduler": [], #
            "Avg. Time / Epoch": [], #
            "Image dimension": [32], #
            "Loss": [], #
            "Min. loss": [], #
            "Accuracy": [],
            "Dataset": [datapath], #
            "Device": [self.device], #
            "Fully connected layers": [self.model.FCS], #
            "Convolutional layers": [self.model.CONVS], #
            "Max pooling layers": [self.model.MAXPOOLS], #
            "Mean pooling layers": [self.model.MEANPOOLS], #
            "Dropout layers": [self.model.DROPOUTS], #
            "Created by": ["Stationær"], #
            "Total training time": [], #
            "Gamma": [self.gamma], #
            "Weight decay": [self.weight_decay], #
            "Model": [self.model.__class__.__name__] #
        }

        # Load data
        self._load_data(datapath)

        # Functions
        self.lossfunction = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.start_lr, weight_decay=self.weight_decay)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.start_lr, weight_decay=self.weight_decay, momentum=self.momentum)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=min_lr)
        # self.scheduler = "AliLR"

        if self.scheduler == "AliLR":
            self.milestep = [i for i in range(math.ceil(epochs/10), epochs + 1, math.ceil(epochs/10))]

        self.data["Loss function"] = [self.lossfunction.__class__.__name__]
        self.data["Optimizer"] = [self.optimizer.__class__.__name__]
        self.data["Scheduler"] = [self.scheduler] if isinstance(self.scheduler, str) else [self.scheduler.__class__.__name__]

    def _get_lr(self, optimizer) -> float:
        for g in optimizer.param_groups:
            return g['lr']

    def _set_lr(self, optimizer, new_lr) -> None:
        for g in optimizer.param_groups:
            g['lr'] = new_lr

    def _print_eta(self, idx, times: list, epoch: int, start, stepend) -> float:
        if idx % 30 == 0 and idx > 0: # Print every 30 steps
            avg_steptime = sum(times) / len(times)
            remaining_steps_current_epoch = self.total_step - idx
            remaining_steps_total = remaining_steps_current_epoch + self.total_step * (self.epochs - epoch - 1)
            eta = avg_steptime * remaining_steps_current_epoch

            # Calculate drift factor
            actual_time = stepend - start
            estimated_time = idx * avg_steptime
            drift_factor = actual_time / estimated_time if estimated_time > 0 else 1

            # Adjust ETA by drift factor
            eta *= drift_factor
            total_eta = avg_steptime * remaining_steps_total * drift_factor

            print(f" {idx}/{self.total_step} | ETA: {eta:.0f}s | Total ETA: {total_eta:.0f}s         ", end="\r")

    def train(self) -> None:
        try:
            self.total_step = len(self.train_loader)
            self.times = []

            self.losses = []
            self.train_losses = []
            self.test_losses = []

            for epoch in range(self.epochs):
                start = perf_counter()
                steptimes = []

                if self.scheduler == "AliLR":
                    if self._get_lr(self.optimizer)*self.gamma > self.min_lr:
                        self._set_lr(self.optimizer, self._get_lr(self.optimizer)*self.gamma)
                    else:
                        self._set_lr(self.optimizer, self.min_lr)

                for i, (images, labels) in enumerate(self.train_loader):
                    stepstart = perf_counter()
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.lossfunction(outputs, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Scheduler step
                    if self.scheduler != "None" and self.scheduler != "AliLR":
                        self.scheduler.step()
                    
                    stepend = perf_counter()
                    steptimes.append(stepend - stepstart)

                    # Calculate Epoch ETA and Total ETA and print
                    self._print_eta(idx=i, times=steptimes, epoch=epoch, start=start, stepend=stepend)

                end = perf_counter()
                measure = end - start
                self.times.append(measure)
                self.losses.append(round(loss.item(), 4))

                # training error/loss
                self.train_losses.append(self.losses[-1])

                # Testing phase per epoch
                self.model.eval() # Set model to evaluation mode
                test_loss = 0
                with torch.no_grad():
                    for images, labels in self.test_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(images)
                        test_loss += self.lossfunction(outputs, labels).item()
                
                self.test_losses.append(test_loss / len(self.test_loader))
                self.model.train()

                # Save latest learning rate
                if self.scheduler != "None" and self.scheduler != "AliLR":
                    self.latest_lr = self.scheduler.get_last_lr()[0]
                elif self.scheduler == "AliLR":
                    self.latest_lr = self._get_lr(self.optimizer)
                else:
                    self.latest_lr = self.start_lr
                
                print(f"Epoch [{epoch+1}/{self.epochs}] | Loss: {self.losses[-1]} | Time elapsed: {measure:.2f}s | Learning rate: {self.latest_lr}")
            
            # Data
            ct = datetime.datetime.now()
            self.data["Date & Time"] = [f"{ct.year}-{ct.month}-{ct.day} {ct.hour}:{ct.minute}:{ct.second}"]
            self.data['Total training time'] = [round(sum(self.times), 1)]
            self.data["Avg. Time / Epoch"] = [round(sum(self.times)/len(self.times), 1)]
            self.data["End LR"] = [self.latest_lr]
            self.data["Loss"] = [self.losses[-1]]
            self.data["Min. loss"] = [min(self.losses)]
        except KeyboardInterrupt:
            print("Training interrupted, some data may be incomplete.")

    def test(self, test_name: str) -> None:
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            self.accuracy = correct/total
        
        # Data
        self.data["Accuracy"] = [self.accuracy]
        print("Done testing. Accuracy:", self.accuracy)
        self.test_name = test_name if test_name else ""

    def _worker(self, process: int, total: int, test: bool, save: bool, model_path: str, excel_path: str) -> None:
        print(f"Worker {process}/{total} on PID {os.getpid()}")
        self.train()
        if test:
            self.test()
        if save:
            self.save_model(f"models/{model_path}")
            self.save_excel(f"Excel/{process}_{excel_path}.xlsx")

    def repeat_train(self, total: int, max_processes: int, delay: int, test: bool, save: bool, model_path: str, excel_path: str) -> None:
        processes = []
        finished, current = 0

        while finished < total:
            if len(processes) < max_processes and current < total:
                current += 1
                p = multiprocessing.Process(target=self._worker, args=(current, total, test, save, model_path, excel_path))
                p.start()
                processes.append(p)
                time.sleep(delay)  # Add delay before starting next process

            for p in processes:
                if not p.is_alive():
                    processes.remove(p)
                    finished += 1
                    print("Finished worker", finished)

            if not processes:
                break

        print("Finished all processes")

    def load_model(self, file_path: str) -> None:
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint)

        # for param in self.model.parameters():
        #    param.requires_grad = False

    def _str_to_filename(self, string: str):
        invalid_chars = ['\\', ':', '?', '<', '>', '|']
        for char in invalid_chars:
            string = string.replace(char, '_')
        return string

    def save_model(self, save_path) -> None:
        customname = f'{self.data["Date & Time"][0]} l{self.data["Loss"][0]} a{self.accuracy:.1f} {self.data["Loss function"][0]}-{self.data["Optimizer"][0]}-{self.data["Scheduler"][0]}'
        path = f"{self._str_to_filename(save_path)}/"

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.model.state_dict(), path+f"{self._str_to_filename(customname)}.pt")
        print("Saved model to", path)

    def save_excel(self, path) -> None:
        while True:
            try:
                try:
                    df = pd.read_excel("Excel/"+path)
                except FileNotFoundError:
                    df = pd.DataFrame(columns=self.data.keys())
                    df.to_excel("Excel/"+path, index=False)
                    print("Created new excel file")
                
                df = pd.concat([df, pd.DataFrame(self.data)], ignore_index=True)

                # Output to excel
                df.to_excel("Excel/"+path, index=False)
                break
            except PermissionError:
                print("Please close the excel file. Retrying in 2 seconds...")
                time.sleep(2)
        print("Wrote to Excel")

    def _load_data(self, root) -> None:
        all_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5],
                                                          std=[0.5])
                                    ])

        classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        self.train_dataset = SubfoldersDataset(root=f'data/{root}/train', filetype="jpg", classes=classes, transform=all_transforms)

        self.test_dataset = SubfoldersDataset(root=f'data/{root}/test', filetype="jpg", classes=classes, transform=all_transforms)

        self.train_loader = torch.utils.data.DataLoader(dataset = self.train_dataset,
                                                batch_size = self.batch_size,
                                                shuffle = True)

        self.test_loader = torch.utils.data.DataLoader(dataset = self.test_dataset,
                                                batch_size = self.batch_size,
                                                shuffle = True)

    def plot_trainvstestloss(self, save_plot: bool = True, display_plot: bool = False) -> None:
        # Plotting the training and testing losses
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.epochs + 1), self.train_losses, label='Training Loss')
        plt.plot(range(1, self.epochs + 1), self.test_losses, label='Testing Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs Testing Loss')
        plt.legend()
        plt.grid(True)

        customname = f'{self.data["Date & Time"][0]} l{self.data["Loss"][0]} a{self.accuracy:.1f} {self.data["Loss function"][0]}-{self.data["Optimizer"][0]}-{self.data["Scheduler"][0]}'
        if display_plot:
            plt.show()
        if save_plot:
            path = f"Images/{self.test_name}"

            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(self._str_to_filename(f'{path}/training_testing_loss_plot {customname}.png'))  # Save the plot as a PNG file

if __name__ == "__main__":
    modelhandler = ModelHandler(
        model =        EmotionRecognizerV2,
        batch_size =   128,
        start_lr =     0.01,
        epochs =       10,
        gamma =        0.5,
        weight_decay = 0.005,
        min_lr =       0,
        momentum =     0.9
    )

    modelhandler.train()
    test = "testing"
    modelhandler.test(test_name=test)
    modelhandler.save_model(save_path="")
