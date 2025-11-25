# train_lbph.py
import cv2
import os
import numpy as np
import pickle
from pathlib import Path

dataset_dir = "dataset"
model_dir = "models"
Path(model_dir).mkdir(parents=True, exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith(("png","jpg","jpeg")):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # optionally detect face and crop (images already cropped by capture script)
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
            if len(faces) > 0:
                x_train.append(img)
                y_labels.append(id_)

# convert to numpy arrays and train
if len(x_train) == 0:
    print("[ERROR] No training images found. Run capture_dataset.py first.")
    exit(1)

print(f"[INFO] Training on {len(x_train)} images for {len(label_ids)} labels.")
recognizer.train(x_train, np.array(y_labels))

model_path = os.path.join(model_dir, "lbph_model.yml")
recognizer.write(model_path)
with open(os.path.join(model_dir, "labels.pickle"), "wb") as f:
    pickle.dump(label_ids, f)

print(f"[INFO] Model saved to {model_path}")
print(f"[INFO] Labels saved to {os.path.join(model_dir, 'labels.pickle')}")
