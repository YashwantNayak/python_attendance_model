# process_video.py
import cv2
import os
import sys
import pickle
import pandas as pd
from datetime import datetime
from pathlib import Path

def process_video(video_path, model_dir="models", output_dir="output", 
                  processed_dir="processed_videos", confidence_threshold=70, 
                  frame_skip=5):
    """
    Video ko process karke face recognition karta hai
    
    Args:
        video_path: Input video file path
        model_dir: Trained model directory
        output_dir: CSV output directory
        processed_dir: Processed video save directory
        confidence_threshold: Recognition confidence threshold (lower = stricter)
        frame_skip: Kitne frames skip karne hain (processing speed ke liye)
    """
    
    # Check if model exists
    model_path = os.path.join(model_dir, "lbph_model.yml")
    labels_path = os.path.join(model_dir, "labels.pickle")
    
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        print("[ERROR] Model files nahi mile. Pehle train_lbph.py run karo.")
        print(f"Required: {model_path} and {labels_path}")
        return None
    
    # Load model
    print("[INFO] Loading face recognition model...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    
    with open(labels_path, "rb") as f:
        orig_labels = pickle.load(f)
    labels = {v: k for k, v in orig_labels.items()}
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Open video
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file nahi mila: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Video open nahi hua: {video_path}")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[INFO] Video info: {width}x{height} @ {fps} FPS, Total frames: {total_frames}")
    
    # Prepare output video
    os.makedirs(processed_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = os.path.join(processed_dir, f"{video_name}_processed_{timestamp}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Track recognized students (set to avoid duplicates)
    recognized_students = set()
    recognition_counts = {}  # Count how many times each student was detected
    
    frame_count = 0
    processed_count = 0
    
    print(f"[INFO] Processing video... (checking every {frame_skip} frames)")
    print("[INFO] Press Ctrl+C to stop early")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for faster processing
            if frame_count % frame_skip != 0:
                out.write(frame)
                continue
            
            processed_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (200, 200))
                
                id_, conf = recognizer.predict(roi_gray)
                name = "Unknown"
                
                if conf < confidence_threshold:
                    name = labels.get(id_, "Unknown")
                    
                    if name != "Unknown":
                        recognized_students.add(name)
                        recognition_counts[name] = recognition_counts.get(name, 0) + 1
                
                # Draw rectangle and name on frame
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name} ({int(conf)})", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Write frame
            out.write(frame)
            
            # Progress update
            if processed_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"[PROGRESS] {progress:.1f}% - Students found: {len(recognized_students)}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Processing stopped by user")
    
    finally:
        cap.release()
        out.release()
    
    print(f"\n[SUCCESS] Processed video saved: {output_video_path}")
    print(f"[INFO] Total frames: {frame_count}, Processed: {processed_count}")
    print(f"[INFO] Unique students recognized: {len(recognized_students)}")
    
    # Save attendance to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"{video_name}_attendance_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    if recognized_students:
        # Create DataFrame with student names and detection counts
        attendance_data = []
        for student in sorted(recognized_students):
            attendance_data.append({
                "Student Name": student,
                "Status": "Present",
                "Detection Count": recognition_counts.get(student, 0),
                "Video": os.path.basename(video_path),
                "Processed On": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        df = pd.DataFrame(attendance_data)
        df.to_csv(csv_path, index=False)
        
        print(f"\n[SUCCESS] Attendance CSV saved: {csv_path}")
        print(f"\n{'='*60}")
        print("ATTENDANCE SUMMARY:")
        print(f"{'='*60}")
        for idx, row in df.iterrows():
            print(f"{idx+1}. {row['Student Name']} - Detected {row['Detection Count']} times")
        print(f"{'='*60}")
        print(f"Total Present: {len(recognized_students)}")
    else:
        print("\n[WARNING] No students recognized in video!")
        print("Possible reasons:")
        print("  1. Model not trained properly (run train_lbph.py)")
        print("  2. No faces detected in video")
        print("  3. Confidence threshold too strict (current: {})".format(confidence_threshold))
    
    return {
        "processed_video": output_video_path,
        "csv_output": csv_path if recognized_students else None,
        "students_found": len(recognized_students),
        "student_list": sorted(list(recognized_students))
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_video.py <video_file_path> [confidence_threshold] [frame_skip]")
        print("Example: python process_video.py uploaded_videos/class_20241125.mp4 70 5")
        print("\nArguments:")
        print("  video_file_path       : Path to video file")
        print("  confidence_threshold  : Recognition threshold (default: 70, lower = stricter)")
        print("  frame_skip           : Process every Nth frame (default: 5, higher = faster)")
        sys.exit(1)
    
    video_path = sys.argv[1]
    confidence = int(sys.argv[2]) if len(sys.argv) > 2 else 70
    skip = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    result = process_video(video_path, confidence_threshold=confidence, frame_skip=skip)
    
    if result and result['students_found'] > 0:
        print(f"\n[DONE] Attendance marked for {result['students_found']} students!")
    else:
        print("\n[DONE] Processing complete but no students found.")
