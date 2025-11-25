# Face Attendance System using LBPH

## Project Structure
- `dataset/` - Images per person (captured using capture_dataset.py)
- `models/` - Trained LBPH model and labels
- `capture_dataset.py` - Capture images from webcam
- `train_lbph.py` - Train LBPH model from dataset
- `recognize_lbph.py` - Recognize faces and mark attendance

## Usage
1. Run `capture_dataset.py` to capture face images
2. Run `train_lbph.py` to train the model
3. Run `recognize_lbph.py` for attendance marking
