import cv2, time, os
import torch
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from models import *
from PIL import Image


class EmotionCamera:
    def __init__(self, model, saved_model_path: str, emotion_labels: str) -> None:
        self.saved_model_path = saved_model_path
        self.emotion_labels = emotion_labels
        self.model = model(num_classes=len(emotion_labels))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(saved_model_path, map_location=self.device))
        self.model.eval()

        os.makedirs('preprocessed_images', exist_ok=True)

    def start(self) -> None:
        capture = cv2.VideoCapture(0)

        self.og_size = [
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ]

        image_counter = 0
        last_frame_time = time.time()

        frame_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
            transforms.RandomRotation(10),      # Randomly rotate images by up to 10 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Randomly change brightness, contrast, and saturation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        preprocessed_images = []
        
        while True:
            # Capture frame from the webcam
            ret, frame = capture.read()
            cv2.imshow('Emotion Recognizer', frame)
            

            if time.time() - last_frame_time >= 1:
                # Preprocess the frame
                grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(grayscale, (48, 48))
                normalized = resized / 255.0
                frame_tensor = torch.from_numpy(normalized)
                frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # Add extra dimensions for batch size and channels
                frame_tensor = frame_tensor.float()
                #print("Preprocessed frame: ", frame_tensor.shape)

                preprocessed_images.append(f"image_{image_counter}.png")
                cv2.imwrite(f'preprocessed_images/image_{image_counter}.png', normalized * 255)
                image_counter += 1

                # Predict emotion
                with torch.no_grad():
                    predictions = self.model(frame_tensor)
                
                # Get predicted emotion on frame
                emotion = torch.argmax(predictions).item()
                emotion_text = self.emotion_labels[emotion]
                print(emotion_text)
                # Display on frame
                cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if len(preprocessed_images) >= 10:
                    # Delete first index image
                    os.remove(f"preprocessed_images/{preprocessed_images.pop(0)}")
                    
                # Update last frame time
                last_frame_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Emotion Recognizer', cv2.WND_PROP_VISIBLE) < 1:
                break
        
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = EmotionCamera(
        model=EmotionRecognizerV2,
        saved_model_path="models/Best model/2024-1-8 17_5_32 l1.2416 a0.6 CrossEntropyLoss-Adam-None_lowest_loss 100V2.pt",
        emotion_labels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    )

    app.start()