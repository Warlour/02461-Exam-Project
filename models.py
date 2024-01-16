import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionRecognizerV1(nn.Module):
    def __init__(self, num_classes, rgb=False):
        '''
        param num_classes: The number of classes to predict
        param rgb: Whether to use rgb or grayscale
        '''
        self.FCS = 2
        self.CONVS = 4
        self.MAXPOOLS = 2
        self.MEANPOOLS = 0
        self.DROPOUTS = 0

        super(self.__class__, self).__init__()
        # in_channels: color channels, black and white is 1, rgb is 3
        self.conv_layer1 = nn.Conv2d(in_channels=3 if rgb else 1, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        '''Fully connected layers tilknytter hver neuron til næste neuron'''
        self.fc1 = nn.Linear(5184, 64) # Fully connected layer
        self.relu1 = nn.ReLU() # Aktiveringsfunktion
        self.fc2 = nn.Linear(64, num_classes)
    
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

# Simple model
class SimpleEmotionRecognizer(nn.Module):
    def __init__(self, num_classes, rgb=False):
        '''
        param num_classes: The number of classes to predict
        param rgb: Whether to use rgb or grayscale
        '''
        self.FCS = 2
        self.CONVS = 2
        self.MAXPOOLS = 2
        self.MEANPOOLS = 0
        self.DROPOUTS = 1

        super(self.__class__, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3 if rgb else 1, out_channels=16, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4608, 64)  # Adjusted to 1152
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.max_pool1(out)
        out = self.conv_layer2(out)
        out = self.max_pool2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class EmotionRecognizerV2(nn.Module):
    def __init__(self, num_classes, rgb=False):
        '''
        param num_classes: The number of classes to predict
        param rgb: Whether to use rgb or grayscale
        '''
        self.FCS = 2
        self.CONVS = 4
        self.MAXPOOLS = 2
        self.MEANPOOLS = 0
        self.DROPOUTS = 3

        super(self.__class__, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3 if rgb else 1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # Batch normalization
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)  # Dropout

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)  # Batch normalization
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)  # Batch normalization
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)  # Dropout

        self.fc1 = nn.Linear(9216, 64)
        self.dropout3 = nn.Dropout(0.5)  # Dropout
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv_layer1(x)))
        x = self.max_pool1(F.relu(self.bn2(self.conv_layer2(x))))
        x = self.dropout1(x)

        x = F.relu(self.bn3(self.conv_layer3(x)))
        x = self.max_pool2(F.relu(self.bn4(self.conv_layer4(x))))
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

class EmotionRecognizerV3(nn.Module):
    def __init__(self, num_classes, rgb=False):
        '''
        param num_classes: The number of classes to predict
        param rgb: Whether to use rgb or grayscale
        '''
        self.FCS = 2
        self.CONVS = 4
        self.MAXPOOLS = 2
        self.MEANPOOLS = 0
        self.DROPOUTS = 3
        
        super(self.__class__, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3 if rgb else 1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Batch normalization
        self.conv_layer2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)  # Batch normalization
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)  # Dropout

        self.conv_layer3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)  # Batch normalization
        self.conv_layer4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)  # Batch normalization
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)  # Dropout

        self.fc1 = nn.Linear(4608, 64)
        self.dropout3 = nn.Dropout(0.5)  # Dropout
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv_layer1(x)))
        x = self.max_pool1(F.relu(self.bn2(self.conv_layer2(x))))
        x = self.dropout1(x)

        x = F.relu(self.bn3(self.conv_layer3(x)))
        x = self.max_pool2(F.relu(self.bn4(self.conv_layer4(x))))
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

class EmotionRecognizerV4(nn.Module):
    def __init__(self, num_classes, rgb=False):
        '''
        param num_classes: The number of classes to predict
        param rgb: Whether to use rgb or grayscale
        '''
        self.FCS = 2
        self.CONVS = 8
        self.MAXPOOLS = 4
        self.MEANPOOLS = 0
        self.DROPOUTS = 5
        
        super(self.__class__, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.fc1 = nn.Linear(1024, 2048)  # The input size depends on the output of your last conv layer
        self.fc2 = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.dropout(x, 0.25)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.dropout(x, 0.25)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.bn3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.dropout(x, 0.25)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.bn4(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.dropout(x, 0.5)

        x = torch.flatten(x, 1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = self.fc2(x)
        return x

class EmotionRecognizerV5(nn.Module):
    def __init__(self, num_classes, rgb=False):
        '''
        param num_classes: The number of classes to predict
        param rgb: Whether to use rgb or grayscale
        '''
        self.FCS = 2
        self.CONVS = 8
        self.MAXPOOLS = 4
        self.MEANPOOLS = 0
        self.DROPOUTS = 5

        super(self.__class__, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(4608, 4096)  # The input size depends on the output of your last conv layer
        self.fc2 = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, 0.25)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, 0.25)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.bn3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, 0.25)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.bn4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, 0.5)

        x = torch.flatten(x, 1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = self.fc2(x)
        return x
    
class EmotionRecognizerV5_2(nn.Module):
    def __init__(self, num_classes, rgb=False):
        '''
        param num_classes: The number of classes to predict
        param rgb: Whether to use rgb or grayscale
        '''
        self.FCS = 2
        self.CONVS = 8
        self.MAXPOOLS = 4
        self.MEANPOOLS = 0
        self.DROPOUTS = 5

        super(self.__class__, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=2, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv7 = nn.Conv2d(256, 512, kernel_size=2, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(8192, 4096)  # Update the input size to match the output size of self.conv8
        self.fc2 = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, 0.25)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, 0.25)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.bn3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, 0.25)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.bn4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.dropout(x, 0.5)

        x = torch.flatten(x, 1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = self.fc2(x)
        return x
    
class EmotionRecognizerV6(nn.Module):
    def __init__(self, num_classes, rgb=False):
        '''
        param num_classes: The number of classes to predict
        param rgb: Whether to use rgb or grayscale
        '''
        self.FCS = 2
        self.CONVS = 6
        self.MAXPOOLS = 3
        self.MEANPOOLS = 0
        self.DROPOUTS = 4

        super(self.__class__, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(6400, 4096)
        self.fc2 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.dropout(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.bn3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = EmotionRecognizerV4(num_classes=7)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)