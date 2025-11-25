# recognize_lbph.py
import cv2
import pickle
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

model_dir = "models"
model_path = os.path.join(model_dir, "lbph_model.yml")
labels_path = os.path.join(model_dir, "labels.pickle")
attendance_csv = "attendance.csv"

if not os.path.exists(model_path) or not os.path.exists(labels_path):
    print("[ERROR] Model or labels not found. Run train_lbph.py first.")
    exit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
with open(labels_path, "rb") as f:
    orig_labels = pickle.load(f)
# invert labels: id -> name
labels = {v:k for k,v in orig_labels.items()}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit(1)

# load existing attendance and seen_today set
if os.path.exists(attendance_csv):
    df = pd.read_csv(attendance_csv)
else:
    df = pd.DataFrame(columns=["timestamp","name"])

today = datetime.now().strftime("%Y-%m-%d")
seen_today = set(df[df['timestamp'].str.startswith(today)]['name'].tolist())

print("[INFO] Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (200,200))
        id_, conf = recognizer.predict(roi_gray)  # conf: lower is better (distance)
        name = "Unknown"
        # tune threshold: smaller -> stricter. LBPH conf typically 0-100
        if conf < 70:
            name = labels.get(id_, "Unknown")
            if name not in seen_today:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_row = pd.DataFrame([{"timestamp": now, "name": name}])
                df = pd.concat([df, new_row], ignore_index=True)
                seen_today.add(name)
                print(f"[INFO] Marked present: {name} at {now}")

        # draw
        color = (0,255,0) if name != "Unknown" else (0,0,255)
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} ({int(conf)})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Recognition LBPH", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# save attendance
df.to_csv(attendance_csv, index=False)
print(f"[INFO] Attendance saved to {attendance_csv}")

cap.release()
cv2.destroyAllWindows()
