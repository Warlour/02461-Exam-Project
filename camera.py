from models import *

class EmotionCamera:
    def __init__(self, model, saved_model_path: str, emotion_labels: str) -> None:
        self.saved_model_path = saved_model_path
        self.emotion_labels = emotion_labels
        self.model = model(num_classes=len(emotion_labels))
    
    def __load_model(self) -> None:
        
    
    def start(self) -> None:
        pass

if __name__ == "__main__":
    app = EmotionCamera(
        model=EmotionRecognizerV2,
        model_path="models/Best model/2024-1-8 17_5_32 l1.2416 a0.6 CrossEntropyLoss-Adam-None_lowest_loss 100V2.pt",
        emotion_labels=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    )