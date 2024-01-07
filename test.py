import cv2
import torch
from models import *
import torch.nn.functional as F

# Load the trained model
model = EmotionRecognizerV2()
model.load_state_dict(torch.load('b64-e20 CEL-ADAM-None/2024-1-4 21.33.57 b64-e20-a96.9173 CEL-ADAM-None.pt'))
model.eval()

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)

while True:
    try:
        # Capture frame from the webcam
        ret, frame = cap.read()

        # Preprocess the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (32, 32))
        normalized_frame = resized_frame / 255.0
        frame_tensor = torch.unsqueeze(torch.from_numpy(normalized_frame), 0)
        frame_tensor = frame_tensor.float()

        # Perform forward pass and get the predictions
        with torch.no_grad():
            predictions = model(frame_tensor)

        # Display the frame with the predicted emotion
        emotion = torch.argmax(predictions).item()
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_text = emotion_labels[emotion]
        cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Emotion Recognition', frame)

        # Wait for 5 seconds before capturing the next frame
        cv2.waitKey(5000)
    except KeyboardInterrupt:
        break

# Release the VideoCapture object and close any open windows
cap.release()
cv2.destroyAllWindows()
