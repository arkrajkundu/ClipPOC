import cv2
import os

def extract_frames(video_path, frame_interval=20, output_dir="extracted_frames"):
    """
    Extracts every 20th frame from the video and saves it.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    frame_files = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_files.append(frame_path)

        frame_idx += 1

    cap.release()
    print(f"Extracted {len(frame_files)} frames.")
    return frame_files
  
video_path = "your_video.mp4"
frame_files = extract_frames(video_path, frame_interval=20, output_dir="your_video_frames")