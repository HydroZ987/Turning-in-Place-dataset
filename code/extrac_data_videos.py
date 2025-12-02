import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---
# Replace 'path/to/your/videos' with the path to your videos folder
# Absolute paths (modify if needed)
BASE_DIR = Path(__file__).resolve().parent.parent  # Remonter un niveau
VIDEO_FOLDER = str(BASE_DIR / 'Videos' / 'Videos')
OUTPUT_FOLDER = str(BASE_DIR / 'data' / 'features')
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# List of 33 landmarks from MediaPipe Pose (for reference)
# Landmarks: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, etc.


def extract_pose_features(video_path, output_csv_path):
    """
    Extract pose coordinates (x, y, z, visibility) from each frame of a video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    # Use Pose model (optimisé GPU)
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 1: équilibre vitesse/précision optimal avec GPU
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        # Prepare data structure
        all_frame_data = []
        frame_number = 0

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break  # End of video

            # Convert BGR to RGB (MediaPipe prefers RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            # Process the image
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # Extract coordinates for the 33 landmarks
                row_data = {'frame_id': frame_number}

                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    # Each landmark has x, y, z, and visibility
                    prefix = f'lm_{idx}'
                    row_data[f'{prefix}_x'] = landmark.x
                    row_data[f'{prefix}_y'] = landmark.y
                    row_data[f'{prefix}_z'] = landmark.z
                    row_data[f'{prefix}_vis'] = landmark.visibility

                all_frame_data.append(row_data)

            frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

    # Create and save DataFrame
    if all_frame_data:
        df = pd.DataFrame(all_frame_data)
        df.to_csv(output_csv_path, index=False)
        print(f"Success: Saved {len(df)} frames to {output_csv_path}")
    else:
        print(f"Warning: No pose detected in {video_path}")


# --- Main Loop ---
print(f"Starting pose extraction from folder: {VIDEO_FOLDER}")
video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]

for filename in tqdm(video_files, desc="Processing videos"):
    video_path = os.path.join(VIDEO_FOLDER, filename)
    # Output CSV file will have the same name as the video, but with .csv extension
    output_csv_filename = filename.replace('.mp4', '.csv').replace('.avi', '.csv').replace('.mov', '.csv')
    output_csv_path = os.path.join(OUTPUT_FOLDER, output_csv_filename)

    extract_pose_features(video_path, output_csv_path)

print("Pose extraction completed.")
