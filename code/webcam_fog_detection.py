import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
import collections

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'code' / 'best_fog_detector.keras'

WINDOW_SIZE = 50
N_FEATURES = 99
THRESHOLD = 0.9  # Augmente de 0.8 a 0.9 pour moins de faux positifs
RIGHT_HIP_LM = 24
LEFT_HIP_LM = 23

print("[OK] Chargement du modele...")
FOG_MODEL = load_model(MODEL_PATH)
print("[OK] Modele charge")

mp_pose = mp.solutions.pose
pose_buffer = collections.deque(maxlen=WINDOW_SIZE)

def normalize_features(landmarks):
    """Normalisation identique a l'entraînement."""
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

print("\n[INFO] Demarrage detection sur WEBCAM")
print("[INFO] Appuie sur ESC pour arreter")
print()

cap = cv2.VideoCapture(0)  # 0 = webcam par defaut

if not cap.isOpened():
    print("[ERR] Impossible d'acceder a la webcam")
    exit()

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
        
        fog_probability = 0.0
        fog_detected = False
        
        if results.pose_landmarks:
            normalized_features = normalize_features(results.pose_landmarks.landmark)
            pose_buffer.append(normalized_features)
            
            if len(pose_buffer) == WINDOW_SIZE:
                input_sequence = np.array(list(pose_buffer), dtype=np.float32).reshape(1, WINDOW_SIZE, N_FEATURES)
                fog_probability = float(FOG_MODEL(input_sequence, training=False).numpy()[0][0])
                fog_detected = fog_probability >= THRESHOLD
                
                if fog_detected:
                    fog_count += 1
        
        # Dessiner landmarks
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Afficher infos
        text = f"FoG: {fog_probability:.2%}" if len(pose_buffer) == WINDOW_SIZE else "FoG: Chargement..."
        color = (0, 0, 255) if fog_detected else (0, 255, 0)
        cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f"Frame: {frame_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"FoG Events: {fog_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "ESC pour arreter", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('[WEBCAM] FoG Detection - Real-Time', image)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            print("\n[STOP] Arret par l'utilisateur")
            break

cap.release()
cv2.destroyAllWindows()

ratio = (fog_count / frame_count * 100) if frame_count > 0 else 0
print(f"\n[STATS] Resumé:")
print(f"   Frames traitees: {frame_count}")
print(f"   Evenements FoG: {fog_count}")
print(f"   Ratio FoG: {ratio:.1f}%")
print()
