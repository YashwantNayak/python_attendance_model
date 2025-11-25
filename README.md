# Face Attendance System using LBPH

## Project Structure
```
python_attendance_model/
├── dataset/              # Training images (organized by student name)
├── models/               # Trained LBPH model and labels
├── uploaded_videos/      # Uploaded classroom videos
├── processed_videos/     # Processed videos with face recognition
├── output/               # Attendance CSV files
├── capture_dataset.py    # Capture training images from webcam
├── train_lbph.py         # Train LBPH face recognition model
├── recognize_lbph.py     # Real-time webcam attendance (original)
├── upload_video.py       # Upload classroom video
└── process_video.py      # Process video and generate attendance
```

## Features
- ✅ Face detection and recognition using LBPH algorithm
- ✅ Webcam-based real-time attendance
- ✅ **NEW**: Video-based attendance processing
- ✅ Automatic duplicate removal
- ✅ CSV attendance reports with timestamps
- ✅ Processed video output with labeled faces

---

## Setup Instructions

### 1. Install Dependencies
```bash
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install packages (force binary wheels)
.\.venv\Scripts\python.exe -m pip install --only-binary :all: numpy opencv-contrib-python pandas
.\.venv\Scripts\python.exe -m pip install imutils
```

### 2. Prepare Training Data
```bash
# Capture face images for each student (50-100 images recommended)
python capture_dataset.py Alice 50
python capture_dataset.py Bob 50
python capture_dataset.py Charlie 50
# ... repeat for all students
```

### 3. Train the Model
```bash
python train_lbph.py
```
This will create:
- `models/lbph_model.yml` - Face recognition model
- `models/labels.pickle` - Student name mappings

---

## Usage

### Option 1: Real-time Webcam Attendance (Original)
```bash
python recognize_lbph.py
```
- Opens webcam
- Recognizes faces in real-time
- Saves attendance to `attendance.csv`
- Press `q` to quit

### Option 2: Video-based Attendance (New Workflow)

#### Step 1: Upload Video
```bash
python upload_video.py <path_to_video>

# Example:
python upload_video.py C:\Videos\class_lecture.mp4
```
Video will be saved to `uploaded_videos/` folder.

#### Step 2: Process Video
```bash
python process_video.py <uploaded_video_path> [confidence_threshold] [frame_skip]

# Example (default settings):
python process_video.py uploaded_videos/class_lecture_20241125_143022.mp4

# Example (custom settings):
python process_video.py uploaded_videos/class_lecture_20241125_143022.mp4 80 10
```

**Parameters:**
- `confidence_threshold` (default: 70) - Lower = stricter matching (0-100)
- `frame_skip` (default: 5) - Process every Nth frame (higher = faster, lower = more accurate)

**Output:**
- **Processed Video**: `processed_videos/<video_name>_processed_<timestamp>.mp4`
  - Original video with face detection boxes and names
- **Attendance CSV**: `output/<video_name>_attendance_<timestamp>.csv`
  - Contains: Student Name, Status, Detection Count, Video, Timestamp

---

## CSV Output Format

**Example: `output/class_lecture_attendance_20241125_143022.csv`**
```csv
Student Name,Status,Detection Count,Video,Processed On
Alice,Present,45,class_lecture.mp4,2024-11-25 14:30:22
Bob,Present,38,class_lecture.mp4,2024-11-25 14:30:22
Charlie,Present,52,class_lecture.mp4,2024-11-25 14:30:22
```

---

## Troubleshooting

### No students recognized in video?
1. **Check model training**: Run `train_lbph.py` first
2. **Adjust confidence threshold**: Try higher values (80-100)
3. **Check video quality**: Ensure faces are visible and well-lit
4. **Reduce frame_skip**: Process more frames (use lower value like 2-3)

### Installation issues (numpy/opencv)?
```bash
# Use pre-built wheels
.\.venv\Scripts\python.exe -m pip install --only-binary :all: numpy opencv-contrib-python pandas
```

### Video processing too slow?
- Increase `frame_skip` parameter (e.g., 10 or 15)
- Use lower resolution videos

---

## Advanced Configuration

### Adjusting Recognition Accuracy
Edit `process_video.py`:
- `confidence_threshold`: Lower = stricter (default: 70)
- `frame_skip`: Higher = faster but less accurate (default: 5)
- `minNeighbors` in `detectMultiScale`: Higher = fewer false positives

### Training Tips
- Capture 50-100 images per student
- Vary lighting conditions during capture
- Include different angles and expressions
- Ensure good lighting during video capture

---

## Quick Start Example

```bash
# 1. Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install --only-binary :all: numpy opencv-contrib-python pandas
.\.venv\Scripts\python.exe -m pip install imutils

# 2. Capture training data
python capture_dataset.py Alice 50
python capture_dataset.py Bob 50

# 3. Train model
python train_lbph.py

# 4. Process video
python upload_video.py C:\Videos\class.mp4
python process_video.py uploaded_videos/class_20241125_143022.mp4

# 5. Check results
# - Processed video: processed_videos/
# - Attendance CSV: output/
```
