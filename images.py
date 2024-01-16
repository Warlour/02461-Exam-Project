import cv2, time, os
import torch
import numpy as np

from models import *
import re

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

        self.accuracies = []
        self.emotions_results = {"angry": [0, 0], "disgust": [0, 0], "fear": [0, 0], "happy": [0, 0], "neutral": [0, 0], "sad": [0, 0], "surprise": [0, 0]}

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
        '''
        Starts the camera and predicts the emotion of the user
        '''
        image_folder = "C:/Users/mathi/Pictures/Camera Roll"
        image_files = os.listdir(image_folder)
        image_counter = 0
        preprocessed_images = []
        
        correct = 0
        total = 0        

        for image_file in image_files:
            try:
                # Read image from the folder
                frame = cv2.imread(os.path.join(image_folder, image_file))
                box_frame = self.bounding_box(frame)

                act_emotion = re.sub(r'\d+\.jpg$', '', image_file)
                print("Actual emotion:", act_emotion, end=" | ")

                # Preprocess the frame
                grayscale = cv2.cvtColor(box_frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(grayscale, (48, 48))
                normalized = resized / 255.0
                frame_tensor = torch.from_numpy(normalized)
                frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # Add extra dimensions for batch size and channels
                frame_tensor = frame_tensor.float()

                # Predict emotion
                with torch.no_grad():
                    predictions = self.model(frame_tensor)
                
                # Get predicted emotion on frame
                guessed_emotion = torch.argmax(predictions).item()
                guessed_emotion_text: str = self.emotion_labels[guessed_emotion]
                print(f"Guessed emotion: {guessed_emotion_text}", end=" = ")

                if guessed_emotion_text.lower() == act_emotion:
                    print("Correct", end="                                                                  \r")
                    correct += 1
                    self.emotions_results[act_emotion][0] += 1
                else:
                    print("Wrong", end="                                                                    \r")
                
                total += 1
                self.emotions_results[act_emotion][1] += 1

                image_counter += 1
                img_path = f"image_{guessed_emotion_text}{image_counter}.png"
                preprocessed_images.append(img_path)
                cv2.imwrite(os.path.join('preprocessed_images', img_path), normalized * 255)

                if len(preprocessed_images) >= 100:
                    # Delete first index image
                    os.remove(os.path.join('preprocessed_images', preprocessed_images.pop(0)))

                cv2.imshow('Emotion Recognizer', frame)
                cv2.imshow('Bounding box', box_frame)

                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Emotion Recognizer', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                print("No face detected", end="                                                        \r")
            
        cv2.destroyAllWindows()

        print(f"{correct}/{total}: {round(100*(correct/total), 4)}%                                                        ")
        self.accuracies.append(correct/total)

if __name__ == "__main__":
    pt_dir = "C:/Users/mathi/OneDrive - Danmarks Tekniske Universitet/Skole/02461 Introduction to Intelligent Systems Fall 23/Eksamen/Modeller og data (do not edit)/Optimering"
    pt_path = "2024-1-11 17_47_4.pt"
    app = EmotionCamera(
        model=EmotionRecognizerV6,
        saved_model_path=os.path.join(pt_dir, pt_path),
        emotion_labels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    )

    iters = 30
    for i in range(iters):
        app.start()
        print(f"{i+1}/{iters}")

    avg_acc = round(sum(app.accuracies)/len(app.accuracies), 4)
    mean = np.mean(app.accuracies)
    std_dev = np.std(app.accuracies)

    confidence = 0.95
    margin_of_error = 1.96 * (std_dev / np.sqrt(len(app.accuracies)))
    conf_int = [round((mean - margin_of_error)*100, 4), round((mean + margin_of_error)*100, 4)]

    emotions_results = ""
    for idx, (key, value) in enumerate(app.emotions_results.items()):
        if idx == len(app.emotions_results.keys())-1:
            emotions_results += f"{key}: {value[0]}/{value[1]}"
        else:
            emotions_results += f"{key}: {value[0]}/{value[1]}, "

    
    with open("accuracies.txt", "a") as f:
        f.write(f"{app.model.__class__.__name__} {pt_path} | Avg. accuracy: {str(avg_acc)} | Confidence interval from {iters} iterations: {conf_int} | {emotions_results}\n")
