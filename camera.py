import cv2, time, os
import torch
import torchvision.transforms as transforms
from models import *
from PIL import Image

import numpy as np

class EmotionCamera:
    def __init__(self, model, saved_model_path: str, emotion_labels: list) -> None:
        '''
        param: model: The model class to use
        param: saved_model_path: The path to the saved model
        param: emotion_labels: The labels of the emotions used to train model
        '''
        self.saved_model_path = saved_model_path
        self.emotion_labels: list = emotion_labels
        self.model = model(num_classes=len(emotion_labels))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.load_state_dict(torch.load(saved_model_path, map_location=self.device))
        self.model.eval()

        os.makedirs('preprocessed_images', exist_ok=True)

        # Load bounding box model
        self.net = cv2.dnn.readNetFromCaffe('Bounding box/deploy.prototxt', 'Bounding box/res10_300x300_ssd_iter_140000.caffemodel')

        self.test_transforms = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

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

    def _preprocess_frame(self, frame) -> torch.Tensor:
        self.box_frame = self.bounding_box(frame)
        # To grayscale
        grayscale_frame = cv2.cvtColor(self.box_frame, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(grayscale_frame)

        # Preprocess the frame
        transformed_frame = self.test_transforms(pil_image)
        frame_tensor = transformed_frame.unsqueeze(0) # Add extra dimensions for batch size
        frame_tensor = frame_tensor.float()
        return frame_tensor

    def _save_image(self, preprocess_frame: torch.Tensor, img_path: str) -> None:
        image = preprocess_frame.squeeze().cpu().numpy()
        image = ((image + 1) / 2.0) * 255.0  # Denormalize the image
        image = Image.fromarray(image.astype('uint8'))
        image.save(os.path.join('preprocessed_images', img_path))

    def start(self) -> None:
        '''
        Starts the processing of images and predicts the emotion
        '''
        capture = cv2.VideoCapture(0)
        image_counter = 0
        preprocessed_images = []

        last_frame_time = time.time()
        self.latest_guess = ""
        self.box_frame = None

        while True:
            try:
                # Capture frame from the webcam
                ret, frame = capture.read()
                if not ret:
                    break
                

                if time.time() - last_frame_time >= 2:
                    # Preprocess the frame
                    preprocess_frame = self._preprocess_frame(frame)

                    # Predict emotion
                    with torch.no_grad():
                        predictions = self.model(preprocess_frame)
                    
                    # Get predicted emotion on frame
                    guessed_emotion = torch.argmax(predictions).item()
                    guessed_emotion_text: str = self.emotion_labels[guessed_emotion]
                    print(f"Guessed emotion: {guessed_emotion_text}                                                  ", end="\r")

                    self.latest_guess = guessed_emotion_text

                    image_counter += 1
                    img_path = f"image_{guessed_emotion_text}{image_counter}.png"
                    preprocessed_images.append(img_path)
                    self._save_image(preprocess_frame, img_path)

                    if len(preprocessed_images) >= 50:
                        # Delete first index image
                        os.remove(os.path.join('preprocessed_images', preprocessed_images.pop(0)))
                        
                    # Update last frame time
                    last_frame_time = time.time()

                # Display guessed emotion on the camera window
                cv2.putText(frame, f'Emotion: {self.latest_guess}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 2, cv2.LINE_AA)

                cv2.imshow('Emotion Recognizer', frame)
                cv2.imshow('Bounding box', self.box_frame)

                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Emotion Recognizer', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                print("No face detected", end="\r")
            
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pt_dir = "C:/Users/mathi/OneDrive - Danmarks Tekniske Universitet/Skole/02461 Introduction to Intelligent Systems Fall 23/Eksamen/Modeller og data (do not edit)/Optimering"
    pt_path = "2024-1-11 17_47_4.pt"
    app = EmotionCamera(
        model=EmotionRecognizerV6,
        saved_model_path=os.path.join(pt_dir, pt_path),
        emotion_labels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    )

    app.start()