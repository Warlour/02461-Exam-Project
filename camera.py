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

        og_size = [
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
        
        while True:
            # Capture frame from the webcam
            ret, frame = capture.read()
            

            if time.time() - last_frame_time >= 5:
                # Preprocessing of frame
                pil_img = Image.fromarray(frame)
                preprocessed = frame_transform(pil_img)
                preprocessed = preprocessed.unsqueeze(0).to(self.device)

                # Convert to numpy array for display
                preprocessed_np = preprocessed.cpu().numpy()  # Convert to numpy array
                preprocessed_np = np.transpose(preprocessed_np, (0, 2, 3, 1))  # (batch, channel, height, width) -> (batch, height, width, channel)
                preprocessed_np = (preprocessed_np * 255).astype(np.uint8)  # Scale pixels to 0-255
                preprocessed_np = cv2.resize(preprocessed_np[0], (self.og_size[0], self.og_size[1]))
                cv2.imshow('Emotion Recognizer', preprocessed_np[0])

                # Denormalize to save image for saving
                denorm = preprocessed * 0.5 + 0.5 # Mean and std
                save_image(denorm, f'preprocessed_images/image_{image_counter}.png')
                image_counter += 1

                # Predict emotion
                with torch.no_grad():
                    predictions = self.model(preprocessed)
                
                # Get predicted emotion on frame
                emotion = torch.argmax(predictions).item()
                emotion_text = self.emotion_labels[emotion]
                print(emotion_text)
                # Display on frame
                cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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