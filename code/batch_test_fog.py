import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
import collections

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'code' / 'best_fog_detector.keras'
VIDEOS_DIR = BASE_DIR / 'Videos' / 'Videos'

WINDOW_SIZE = 50 
N_FEATURES = 99
RIGHT_HIP_LM = 24
LEFT_HIP_LM = 23

# Charger le modele
print("[OK] Chargement du modele...")
FOG_MODEL = load_model(MODEL_PATH)
print("[OK] Modele charge")

mp_pose = mp.solutions.pose

def normalize_features(landmarks):
    """Applique la normalisation identique a l'entraînement."""
    features = []
    for i in range(33):
        features.extend([landmarks[i].x, landmarks[i].y, landmarks[i].z])
    features = np.array(features)
    
    z_values = [landmarks[i].z for i in range(33)]
    mean_z = np.mean(z_values)
    if mean_z == 0:
        mean_z = 1.0
    
    ref_x = landmarks[RIGHT_HIP_LM].x
    ref_y = landmarks[RIGHT_HIP_LM].y
    
    normalized_features = np.zeros_like(features)
    for i in range(33):
        normalized_features[i*3] = (features[i*3] - ref_x) / mean_z
        normalized_features[i*3+1] = (features[i*3+1] - ref_y) / mean_z
        normalized_features[i*3+2] = features[i*3+2] / mean_z
    
    hip_dist = np.sqrt(
        (landmarks[RIGHT_HIP_LM].x - landmarks[LEFT_HIP_LM].x)**2 +
        (landmarks[RIGHT_HIP_LM].y - landmarks[LEFT_HIP_LM].y)**2
    )
    if hip_dist < 0.05:
        hip_dist = 1.0
    normalized_features /= hip_dist
    
    return normalized_features.tolist()

def test_video(video_path, threshold=0.8):
    """Teste une video et retourne les stats."""
    if not video_path.exists():
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    pose_buffer = collections.deque(maxlen=WINDOW_SIZE)
    fog_count = 0
    frame_count = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            frame_count += 1
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                normalized_features = normalize_features(results.pose_landmarks.landmark)
                pose_buffer.append(normalized_features)
                
                if len(pose_buffer) == WINDOW_SIZE:
                    input_sequence = np.array(list(pose_buffer), dtype=np.float32).reshape(1, WINDOW_SIZE, N_FEATURES)
                    fog_probability = float(FOG_MODEL(input_sequence, training=False).numpy()[0][0])
                    
                    if fog_probability >= threshold:
                        fog_count += 1
    
    cap.release()
    
    fog_ratio = (fog_count / frame_count * 100) if frame_count > 0 else 0
    return {
        'frames': frame_count,
        'fog_events': fog_count,
        'ratio': fog_ratio
    }

# --- Test sur toutes les videos ---
print("\n" + "="*70)
print("TEST BATCH: DETECTION FoG SUR TOUTES LES VIDEOS")
print("="*70)

video_files = sorted(VIDEOS_DIR.glob("*.mp4"))
print(f"\n[INFO] {len(video_files)} videos trouvees")

results = []
for i, video_file in enumerate(video_files[:10], 1):  # Tester les 10 premieres
    print(f"\n[{i}/10] Traitement: {video_file.name}...", end=" ", flush=True)
    
    result = test_video(video_file, threshold=0.8)
    if result:
        results.append({
            'video': video_file.name,
            'frames': result['frames'],
            'fog_events': result['fog_events'],
            'ratio': result['ratio']
        })
        print(f"OK - Frames: {result['frames']}, FoG: {result['fog_events']}, Ratio: {result['ratio']:.1f}%")
    else:
        print("ERREUR")

# --- Afficher le resume ---
print("\n" + "="*70)
print("RESUME DES RESULTATS (THRESHOLD=0.8):")
print("="*70)
print(f"{'Video':<25} {'Frames':>10} {'FoG':>10} {'Ratio':>10}")
print("-"*70)

for res in results:
    print(f"{res['video']:<25} {res['frames']:>10} {res['fog_events']:>10} {res['ratio']:>9.1f}%")

avg_ratio = np.mean([r['ratio'] for r in results]) if results else 0
print("-"*70)
print(f"{'MOYENNE':<25} {'':<10} {'':<10} {avg_ratio:>9.1f}%")
print("="*70)
