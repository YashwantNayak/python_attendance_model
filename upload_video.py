# upload_video.py
import os
import sys
import shutil
from datetime import datetime

def upload_video(source_path, destination_folder="uploaded_videos"):
    """
    Video ko uploaded_videos folder me copy karta hai
    
    Args:
        source_path: Video file ka path
        destination_folder: Destination folder (default: uploaded_videos)
    
    Returns:
        Uploaded file ka path ya None if error
    """
    if not os.path.exists(source_path):
        print(f"[ERROR] File nahi mila: {source_path}")
        return None
    
    # Check if file is video
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    file_ext = os.path.splitext(source_path)[1].lower()
    if file_ext not in video_extensions:
        print(f"[ERROR] Invalid video format. Supported: {', '.join(video_extensions)}")
        return None
    
    # Create destination folder if not exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = os.path.splitext(os.path.basename(source_path))[0]
    dest_filename = f"{original_name}_{timestamp}{file_ext}"
    dest_path = os.path.join(destination_folder, dest_filename)
    
    # Copy file
    try:
        print(f"[INFO] Uploading video: {os.path.basename(source_path)}")
        shutil.copy2(source_path, dest_path)
        file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"[SUCCESS] Video uploaded: {dest_path}")
        print(f"[INFO] File size: {file_size_mb:.2f} MB")
        return dest_path
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upload_video.py <video_file_path>")
        print("Example: python upload_video.py C:\\Videos\\class_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    uploaded_path = upload_video(video_path)
    
    if uploaded_path:
        print(f"\n[NEXT STEP] Process video using:")
        print(f"python process_video.py {uploaded_path}")
