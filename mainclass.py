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
                 datapath: str = "FER2013", name: str = "") -> None:
        # Variables
        self.batch_size = batch_size
        self.classes = 7
        self.start_lr = start_lr
        self.epochs = epochs
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.momentum = momentum
        self.name = name

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
        self.__load_data(datapath)

        # Functions
        self.lossfunction = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.start_lr, weight_decay=self.weight_decay)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.start_lr, weight_decay=self.weight_decay, momentum=self.momentum)

        #self.scheduler = "None"
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=min_lr)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=self.gamma, patience=5, verbose=False)
        self.scheduler = "AliLR"

        if self.scheduler == "AliLR":
            self.milestep = [i for i in range(math.ceil(epochs/10), epochs + 1, math.ceil(epochs/10))]

        self.data["Scheduler"] = [self.scheduler] if isinstance(self.scheduler, str) else [self.scheduler.__class__.__name__]

        self.data["Loss function"] = [self.lossfunction.__class__.__name__]
        self.data["Optimizer"] = [self.optimizer.__class__.__name__]

    def __get_lr(self, optimizer) -> float:
        for g in optimizer.param_groups:
            return g['lr']

    def __set_lr(self, optimizer, new_lr) -> None:
        for g in optimizer.param_groups:
            g['lr'] = new_lr

    def __print_eta(self, idx, times: list, epoch: int, start, stepend) -> float:
        if idx % 30 == 0 and idx > 0: # Print every 30 steps
            avg_steptime = sum(times) / len(times)
            remaining_steps_current_epoch = self.__total_step - idx
            remaining_steps_total = remaining_steps_current_epoch + self.__total_step * (self.epochs - epoch - 1)
            eta = avg_steptime * remaining_steps_current_epoch

            # Calculate drift factor
            actual_time = stepend - start
            estimated_time = idx * avg_steptime
            drift_factor = actual_time / estimated_time if estimated_time > 0 else 1

            # Adjust ETA by drift factor
            eta *= drift_factor
            total_eta = avg_steptime * remaining_steps_total * drift_factor

            print(f" {idx}/{self.__total_step} | ETA: {eta:.0f}s | Total ETA: {total_eta:.0f}s         ", end="\r")

    def train(self, stoppage: bool = False) -> None:
        self.__total_step = len(self.__train_loader)
        self.__times = []

        self.__losses = []
        self.__train_losses = []
        self.__validation_losses = []
        self.__lowest_loss_model = None

        # Early Stoppage
        early_stop_counter = 0
        patience_threshold = 10  # Number of epochs to wait before stopping
        min_validation_loss = float('inf')
        validation_loss = 0
        best_validation_loss = float('inf')

        try:
            for epoch in range(self.epochs):
                start = perf_counter()
                steptimes = []

                if self.scheduler == "AliLR":
                    if self.__get_lr(self.optimizer)*self.gamma > self.min_lr:
                        self.__set_lr(self.optimizer, self.__get_lr(self.optimizer)*self.gamma)
                    else:
                        self.__set_lr(self.optimizer, self.min_lr)

                for i, (images, labels) in enumerate(self.__train_loader):
                    stepstart = perf_counter()
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(images)
                    loss = self.lossfunction(outputs, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Scheduler step
                    if self.scheduler != "None" and self.scheduler != "AliLR" and self.scheduler.__class__.__name__ != "ReduceLROnPlateau":
                        self.scheduler.step()
                    
                    # IF ReduceLROnPlateau
                    if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                        #val_loss =
                        self.scheduler.step(metrics=validation_loss)
                    
                    stepend = perf_counter()
                    steptimes.append(stepend - stepstart)

                    # Calculate Epoch ETA and Total ETA and print
                    self.__print_eta(idx=i, times=steptimes, epoch=epoch, start=start, stepend=stepend)

                end = perf_counter()
                measure = end - start
                self.__times.append(measure)
                self.__losses.append(round(loss.item(), 4))

                # training error/loss
                self.__train_losses.append(self.__losses[-1])

                # Validation phase per epoch
                self.model.eval() # Set model to evaluation mode
                validation_loss = 0
                with torch.no_grad():
                    for images, labels in self.__validation_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        outputs = self.model(images)
                        validation_loss += self.lossfunction(outputs, labels).item()
                
                validation_loss_avg = validation_loss / len(self.__validation_loader)  # Record the validation loss
                self.__validation_losses.append(validation_loss_avg)

                if validation_loss_avg < best_validation_loss:
                    best_validation_loss = validation_loss_avg
                    self.__lowest_loss_model = self.model.state_dict()

                # Early stopping
                if stoppage:
                    if validation_loss_avg < min_validation_loss:
                        min_validation_loss = validation_loss_avg
                        early_stop_counter = 0  # reset counter if test loss decreases
                    else:
                        early_stop_counter += 1  # increment counter if test loss does not decrease

                    if early_stop_counter > patience_threshold:
                        print(f"Early stopping triggered at epoch {epoch+1}               ")
                        break  # Break out of the training loop

                self.model.train()

                # Save latest learning rate
                if self.scheduler != "None" and self.scheduler != "AliLR" and self.scheduler.__class__.__name__ != "ReduceLROnPlateau":
                    self.latest_lr = self.scheduler.get_last_lr()[0]
                elif self.scheduler == "AliLR":
                    self.latest_lr = self.__get_lr(self.optimizer)
                else:
                    self.latest_lr = self.start_lr

                print(f"Epoch [{epoch+1}/{self.epochs}] | Train Loss: {self.__losses[-1]} | Validation Loss: {validation_loss_avg} | Time elapsed: {measure:.2f}s | Learning rate: {self.latest_lr}")

                self.trained = True
        except KeyboardInterrupt:
            print("Training interrupted, use save_model() to save the latest model and lowest loss model.")
        
        # Data
        ct = datetime.datetime.now()
        self.data["Date & Time"] = [f"{ct.year}-{ct.month}-{ct.day} {ct.hour}:{ct.minute}:{ct.second}"]
        self.data['Total training time'] = [round(sum(self.__times), 1)]
        self.data["Avg. Time / Epoch"] = [round(sum(self.__times)/len(self.__times), 1)]
        self.data["End LR"] = [self.latest_lr]
        self.data["Loss"] = [self.__losses[-1]]
        self.data["Min. loss"] = [min(self.__losses)]

    def test(self) -> None:
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.__test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            self.accuracy = correct/total
        self.tested = True

        # Data
        self.data["Accuracy"] = [self.accuracy]
        print("Done testing. Accuracy:", self.accuracy)

    def __worker(self, process: int, total: int, test: bool, save: bool, model_path: str, excel_path: str) -> None:
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
                p = multiprocessing.Process(target=self.__worker, args=(current, total, test, save, model_path, excel_path))
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

    def __str_to_filename(self, string: str):
        invalid_chars = ['\\', ':', '?', '<', '>', '|']
        for char in invalid_chars:
            string = string.replace(char, '_')
        return string

    def save_model(self, save_path, save_lowest: bool = False) -> None:
        self.customname = self.__str_to_filename(f'{self.data["Date & Time"][0]} l{self.data["Loss"][0]} a{self.accuracy:.1f} {self.data["Loss function"][0]}-{self.data["Optimizer"][0]}-{self.data["Scheduler"][0]}')
        path = os.path.join(save_path, self.customname)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.model.state_dict(), f"{path}.pt")
        print("Saved model to", save_path)
        if save_lowest:
            torch.save(self.__lowest_loss_model, f"{path}_lowest_loss {self.name}.pt")
            print("Saved lowest loss model to", save_path)

    def save_excel(self, save_path) -> None:
        '''
        Saves the data to an excel file\\
        Don't add file extension

        param save_path: Directory of the file to save to
        '''
        save_path = os.path.join("Excel", save_path)
        file_path = save_path+".xlsx"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        while True:
            try:
                try:
                    df = pd.read_excel(file_path)
                except FileNotFoundError:
                    df = pd.DataFrame(columns=self.data.keys())
                    df.to_excel(file_path, index=False)
                    print("Created new excel file")
                
                df = pd.concat([df, pd.DataFrame(self.data)], ignore_index=True)

                # Output to excel
                df.to_excel(file_path, index=False)
                break
            except PermissionError:
                print("Please close the excel file. Retrying in 2 seconds...")
                time.sleep(2)
        print("Wrote to Excel")

    def __load_data(self, root) -> None:
        # all_transforms = transforms.Compose([transforms.Resize((32, 32)),
        #                              transforms.ToTensor(),
        #                              transforms.Normalize(mean=[0.5],
        #                                                   std=[0.5])
        #                             ])
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

        # Only basic transforms for the validation dataset
        validation_transforms = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        self.__train_dataset = SubfoldersDataset(root=f'data/{root}/train', filetype="jpg", classes=classes, transform=train_transforms)

        self.__test_dataset = SubfoldersDataset(root=f'data/{root}/test', filetype="jpg", classes=classes, transform=test_transforms)

        self.__validation_dataset = SubfoldersDataset(root=f'data/{root}/validation', filetype="jpg", classes=classes, transform=validation_transforms)

        self.__train_loader = torch.utils.data.DataLoader(dataset = self.__train_dataset,
                                                batch_size = self.batch_size,
                                                shuffle = True)

        self.__test_loader = torch.utils.data.DataLoader(dataset = self.__test_dataset,
                                                batch_size = self.batch_size,
                                                shuffle = True)
        
        self.__validation_loader = torch.utils.data.DataLoader(dataset = self.__validation_dataset,
                                                batch_size = self.batch_size,
                                                shuffle = True)

    def plot_trainvstestloss(self, save_plot: bool = True, display_plot: bool = False, save_path: str = "") -> None:
        if not self.trained and not self.tested:
            print("Please train and test the model first.")
            return
        
        actual_epochs = len(self.__validation_losses)
        # Plotting the training and testing losses
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, actual_epochs + 1), self.__train_losses[:actual_epochs], label='Training Loss')
        plt.plot(range(1, actual_epochs + 1), self.__validation_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True)

        customname = f'{self.data["Date & Time"][0]} l{self.data["Loss"][0]} a{self.accuracy:.1f} {self.data["Loss function"][0]}-{self.data["Optimizer"][0]}-{self.data["Scheduler"][0]}'
        if display_plot:
            plt.show()
        if save_plot:
            path = os.path.join("Images", save_path)

            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(os.path.join("Images", self.__str_to_filename(self.name+" "+customname)+".png"))  # Save the plot as a PNG file

if __name__ == "__main__":
    name = "Test 18ns"
    modelhandler = ModelHandler(
        model =        EmotionRecognizerV2,
        batch_size =   64,
        start_lr =     0.001,
        epochs =       100,
        gamma =        0.5,
        weight_decay = 0.005,
        min_lr =       0,
        momentum =     0.9,
        name=name
    )
    modelhandler.train(stoppage=False)
    modelhandler.test()
    print(name)
    modelhandler.save_model("models/New tests no stoppage/"+name, save_lowest=True)
    modelhandler.save_excel("New tests no stoppage/"+name)
    modelhandler.plot_trainvstestloss(save_path=name)