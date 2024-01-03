import os
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
from torchvision import transforms

class CustomFER2013Dataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(CustomFER2013Dataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images, self.labels = self._load_dataset()

    def _load_dataset(self):
        images = []
        labels = []

        for emotion in self.classes:
            emotion_folder = os.path.join(self.root, emotion)
            for filename in os.listdir(emotion_folder):
                if filename.endswith(".jpg"):
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

# Example usage:
# custom_transforms = transforms.Compose([...])  # Define your transformations here
# C = CustomFER2013Dataset(Vis)
# print(custom_fer2013_dataset)
