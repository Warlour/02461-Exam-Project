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
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2023, 0.1994, 0.2010])])