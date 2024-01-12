import os
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
from torchvision import transforms

class SubfoldersDataset(VisionDataset):
    def __init__(self, root, filetype: str, classes: list, transform=None, target_transform=None, oversample: bool = False):
        super(self.__class__, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.filetype = filetype
        self.images, self.labels = self._load_dataset()

    def _load_dataset(self):
        images = []
        labels = []

        for emotion in self.classes:
            emotion_folder = os.path.join(self.root, emotion)
            for filename in os.listdir(emotion_folder):
                if filename.endswith(f".{self.filetype}"):
                    img_path = os.path.join(emotion_folder, filename)
                    images.append(img_path)
                    labels.append(self.class_to_idx[emotion])

        #print(images)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx], self.labels[idx]
        img = read_image(img_path)
        
        if self.transform:
            trans = transforms.ToPILImage()
            img = self.transform(trans(img))

        if self.target_transform:
            label = self.target_transform(label)

        return img, label
    
class FolderDataset(VisionDataset):
    def __init__(self, root, filetype: str, transform=None, target_transform=None):
        super(self.__class__, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.filetype = filetype
        self.images, self.labels = self._load_dataset()

    def _load_dataset(self):
        images = []
        labels = []

        for filename in os.listdir(self.root):
            if filename.endswith(f".{self.filetype}"):
                img_path = os.path.join(self.root, filename)
                images.append(img_path)
                labels.append(0)

        #print(images)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx], self.labels[idx]
        img = read_image(img_path)
        
        if self.transform:
            trans = transforms.ToPILImage()
            img = self.transform(trans(img))

        if self.target_transform:
            label = self.target_transform(label)

        return img, label