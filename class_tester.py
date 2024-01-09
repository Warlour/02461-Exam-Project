import torch
from torchvision import transforms
from PIL import Image
import os
from models import *


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

percentages = []

for emotion_label in emotion_labels:
    path = f'data/FER2013/test/{emotion_label.lower()}'
    model = EmotionRecognizerV2(num_classes=7)
    model_path = "models/Best model/2024-1-8 17_5_32 l1.2416 a0.6 CrossEntropyLoss-Adam-None 100V2.pt"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
    model.eval()


    correct = 0
    total = 0

    for image_file in os.listdir(path):
        if image_file.endswith('.jpg'):
            image = Image.open(os.path.join(path, image_file))
            transform = transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

            image_tensor = transform(image)
            with torch.no_grad():
                predictions = model(image_tensor.unsqueeze(0))
            
            emotion = torch.argmax(predictions).item()
            emotion_text = emotion_labels[emotion]
            #print(emotion_text)
            if emotion_text == emotion_label:
                correct += 1
            total += 1
    
    p = correct/total
    print(emotion_label, p)
    percentages.append(p)

print(sum(percentages)/len(emotion_labels))
