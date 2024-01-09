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

        # Load bounding box model
        self.net = cv2.dnn.readNetFromCaffe('Bounding box/deploy.prototxt', 'Bounding box/res10_300x300_ssd_iter_140000.caffemodel')

    def bounding_box(self, frame):
        resized_image = cv2.resize(frame, (300, 300))

        # Preprocess image
        blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Feed into deep neural network
        self.net.setInput(blob)
        detections = self.net.forward()

        largest_face = 0
        largest_face_coords = None

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype(int)

                face_size = (endX - startX) * (endY - startY)

                if face_size > largest_face_size:
                    largest_face_size = face_size
                    largest_face_coords = (startX, startY, endX, endY)

        if largest_face_coords is not None:
            (startX, startY, endX, endY) = largest_face_coords
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cropped_face = frame[startY:endY, startX:endX]

            cv2.imshow('Cropped face', cropped_face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No face detected")


    def start(self) -> None:
        capture = cv2.VideoCapture(0)

        self.og_size = [
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ]

        image_counter = 0
        last_frame_time = time.time()

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