import cv2
import time
import torch
from models import EmotionRecognizerV2
import os

# Load the trained model
model = EmotionRecognizerV2(num_classes=7)
model.load_state_dict(torch.load('models/V2/2024-1-7 17_17_47 l1.7896 a0.3 CrossEntropyLoss-Adam-CosineAnnealingLR.pt', map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
model.eval()

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

last_frame_time = time.time()

os.makedirs('preprocessed_images', exist_ok=True)
image_counter = 0

while True:
    # Capture frame from the webcam
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    # Check if 5 seconds have passed since the last frame was processed
    if time.time() - last_frame_time >= 5:
        # Preprocess the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (32, 32))
        normalized_frame = resized_frame / 255.0
        frame_tensor = torch.from_numpy(normalized_frame)
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # Add extra dimensions for batch size and channels
        frame_tensor = frame_tensor.float()
        print("Preprocessed frame: ", frame_tensor.shape)

        cv2.imwrite(f'preprocessed_images/image_{image_counter}.png', resized_frame * 255)
        image_counter += 1

        # Perform forward pass and get the predictions
        with torch.no_grad():
            predictions = model(frame_tensor)
        print(predictions)

        # Display the predicted emotion on the frame
        emotion = torch.argmax(predictions).item()
        emotion_text = emotion_labels[emotion]
        print(emotion_text)
        cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Update the last frame time
        last_frame_time = time.time()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close any open windows
cap.release()
cv2.destroyAllWindows()