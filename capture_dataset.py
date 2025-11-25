# capture_dataset.py
import cv2
import os
import sys

def create_person_dataset(name, samples=50, save_dir="dataset"):
    person_dir = os.path.join(save_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0
    print(f"[INFO] Press 'c' to capture face images for '{name}', 'q' to quit early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(frame, f"Captures: {count}/{samples}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
        cv2.imshow("Capture Dataset", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(faces) == 0:
                print("[WARN] No face detected, try again.")
                continue
            # save first detected face (crop & resize)
            (x,y,w,h) = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200,200))
            img_path = os.path.join(person_dir, f"{name}_{count:03d}.jpg")
            cv2.imwrite(img_path, face_img)
            print(f"[INFO] Saved {img_path}")
            count += 1
            if count >= samples:
                break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python capture_dataset.py <PersonName> [num_samples]")
        sys.exit(1)
    person_name = sys.argv[1]
    samples = int(sys.argv[2]) if len(sys.argv) >=3 else 50
    create_person_dataset(person_name, samples)
