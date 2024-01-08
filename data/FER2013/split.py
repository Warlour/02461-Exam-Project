import os
import shutil
from random import sample

def split_data_for_validation(train_dir, val_dir, validation_split=0.125):
    classes = os.listdir(train_dir)

    for emotion in classes:
        emotion_train_dir = os.path.join(train_dir, emotion)
        emotion_val_dir = os.path.join(val_dir, emotion)
        os.makedirs(emotion_val_dir, exist_ok=True)  # Create class folder in validation directory

        images = os.listdir(emotion_train_dir)
        val_size = int(len(images) * validation_split)
        print(val_size)

        val_images = sample(images, val_size)  # Randomly select images for validation

        for img in val_images:
            src_path = os.path.join(emotion_train_dir, img)
            dest_path = os.path.join(emotion_val_dir, img)
            shutil.move(src_path, dest_path)

train_dir = 'data/FER2013/train'
val_dir = 'data/FER2013/validation'
split_data_for_validation(train_dir, val_dir)