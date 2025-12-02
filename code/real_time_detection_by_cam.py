import cv2
import mediapipe as mp
import numpy as np
import collections
from tensorflow.keras.models import load_model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'best_fog_detector.keras'
VIDEO_FILE_PATH = BASE_DIR / 'Videos' / 'Videos' / 'PDFE01_1.mp4'

WINDOW_SIZE = 50 
N_LANDMARKS = 33
N_FEATURES = N_LANDMARKS * 2  # 66 features (X et Y pour 33 landmarks)
THRESHOLD = 0.8  # Seuil de probabilité pour l'alerte FoG
RIGHT_HIP_LM = 24
LEFT_HIP_LM = 23

try:
    FOG_MODEL = load_model(MODEL_PATH)
    print("✅ Modèle FoG chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur de chargement du modèle: {e}. Vérifiez le chemin : {MODEL_PATH}")
    exit()

mp_pose = mp.solutions.pose
pose_buffer = collections.deque(maxlen=WINDOW_SIZE)

# --- 2. FONCTION DE NORMALISATION (Copie de 04_real_time_detection.py) ---
def normalize_features(landmarks):
    """
    Applique la même logique de normalisation (Centrage + Scaling)
    que celle utilisée pour l'entraînement.
    """
    
    # Collecte et Centrage (Translation)
    ref_x = landmarks[RIGHT_HIP_LM].x
    ref_y = landmarks[RIGHT_HIP_LM].y
    
    current_features = []
    for i in range(N_LANDMARKS):
        current_features.append(landmarks[i].x - ref_x) 
        current_features.append(landmarks[i].y - ref_y)
    
    normalized_features = np.array(current_features)

    # Scaling (Mise à l'échelle)
    hip_dist = np.sqrt(
        (landmarks[RIGHT_HIP_LM].x - landmarks[LEFT_HIP_LM].x)**2 +
        (landmarks[RIGHT_HIP_LM].y - landmarks[LEFT_HIP_LM].y)**2
    )
    
    if hip_dist < 0.05: 
        hip_dist = 1.0 
    
    normalized_features /= hip_dist
    
    return normalized_features.tolist()

# --- 3. BOUCLE DE DÉTECTION SUR FICHIER ---
def test_video_detection():
    cap = cv2.VideoCapture(str(VIDEO_FILE_PATH)) 
    
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la source vidéo à {VIDEO_FILE_PATH}.")
        return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        print(f"Démarrage de la détection sur fichier : {VIDEO_FILE_PATH}")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Fin de la vidéo ou erreur de lecture.")
                break 

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            fog_probability = 0.0
            
            if results.pose_landmarks:
                
                # 1. Normalisation
                normalized_features = normalize_features(results.pose_landmarks.landmark)
                
                # 2. Mise à jour du buffer
                pose_buffer.append(normalized_features)
                
                # 3. Prédiction
                if len(pose_buffer) == WINDOW_SIZE:
                    input_sequence = np.array(pose_buffer, dtype=np.float32).reshape(1, WINDOW_SIZE, N_FEATURES)
                    fog_probability = FOG_MODEL.predict(input_sequence, verbose=0)[0][0]
                    
                    # 4. Affichage
                    if fog_probability >= THRESHOLD:
                        alerte_text = f"!!! ALERTE FOG: {fog_probability:.2f} !!!"
                        couleur = (0, 0, 255) 
                    else:
                        alerte_text = f"FoG Score: {fog_probability:.2f}"
                        couleur = (0, 255, 0)
                        
                    cv2.putText(image, alerte_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, couleur, 2)
            
            # Affichage
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            cv2.imshow('FoG Video Test', image)
            
            # cv2.waitKey(1) est utilisé pour lire la vidéo à une vitesse proche de l'originale
            if cv2.waitKey(1) & 0xFF == 27: break 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_video_detection()