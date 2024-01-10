import matplotlib.pyplot as plt

class plots:
    def __init__(self, labels) -> None:
        self.labels = labels

    def plot_emotion_distribution(self):
        plt.bar(self.labels.keys(), self.labels.values())
        plt.xlabel('Emotions')
        plt.ylabel('Number of samples')
        plt.title('FER2013 Class distribution')
        plt.show()

if __name__ == "__main__":
    labels = {'Angry': 3496, 'Disgust': 382, 'Fear': 3585, 'Happy': 6314, 'Neutral': 4345, 'Sad': 4227, 'Surprise': 2775}
    FER2013plots = plots(labels)
    FER2013plots.plot_emotion_distribution()