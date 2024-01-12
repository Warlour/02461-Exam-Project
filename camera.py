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

        largest_face_size = 0
        largest_face_coords = None

        # Detect faces
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
            #cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cropped_face = frame[startY:endY, startX:endX]

            # cv2.imshow('Cropped face', cropped_face)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return cropped_face
        else:
            print("No face detected")


    def start(self) -> None:
        capture = cv2.VideoCapture(0)

        image_counter = 0
        last_frame_time = time.time()

        preprocessed_images = []
        emotion_text = ""

        while True:
            try:
                # Capture frame from the webcam
                ret, frame = capture.read()
                box_frame = self.bounding_box(frame)

                if time.time() - last_frame_time >= 0.5:
                    # Preprocess the frame
                    grayscale = cv2.cvtColor(box_frame, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(grayscale, (48, 48))
                    normalized = resized / 255.0
                    frame_tensor = torch.from_numpy(normalized)
                    frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # Add extra dimensions for batch size and channels
                    frame_tensor = frame_tensor.float()
                    #print("Preprocessed frame: ", frame_tensor.shape)

                    # Predict emotion
                    with torch.no_grad():
                        predictions = self.model(frame_tensor)
                    
                    # Get predicted emotion on frame
                    emotion = torch.argmax(predictions).item()
                    emotion_text = self.emotion_labels[emotion]
                    print(emotion_text)

                    # Display on frame
                    cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    image_counter += 1
                    img_path = f"image_{emotion_text}{image_counter}.png"
                    preprocessed_images.append(img_path)
                    cv2.imwrite(os.path.join('preprocessed_images', img_path), normalized * 255)

                    if len(preprocessed_images) >= 10:
                        # Delete first index image
                        os.remove(f"preprocessed_images/{preprocessed_images.pop(0)}")
                        
                    # Update last frame time
                    last_frame_time = time.time()

                
                
                cv2.imshow('Emotion Recognizer', frame)
                cv2.imshow('Bounding box', box_frame)

                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Emotion Recognizer', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                print("No face detected", end="\r")
            
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = EmotionCamera(
        model=EmotionRecognizerV4,
        saved_model_path="models/V4/2024-1-9 22_13_23 lowest_loss.pt",
        emotion_labels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    )

    # app.bounding_box(cv2.imread('preprocessed_images/image_23.png'))
    app.start()