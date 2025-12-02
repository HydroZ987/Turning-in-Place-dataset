import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path
import collections

# --- Configuration des chemins ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'code' / 'best_fog_detector.keras'

# ⚠️ Constantes (DOIVENT ÊTRE IDENTIQUES à celles d'entraînement)
WINDOW_SIZE = 50 
N_FEATURES = 99  # 33 landmarks * 3 (X, Y, Z)
THRESHOLD = 0.8  # Seuil de probabilité pour l'alerte FoG
RIGHT_HIP_LM = 24
LEFT_HIP_LM = 23

# --- Initialisation ---
try:
    FOG_MODEL = load_model(MODEL_PATH)
    print("[OK] Modele FoG charge avec succes.")
except Exception as e:
    print(f"[ERR] Erreur de chargement du modele: {e}")
    exit()

mp_pose = mp.solutions.pose
pose_buffer = collections.deque(maxlen=WINDOW_SIZE)


def normalize_features(landmarks):
    """Applique la même logique de normalisation que l'entraînement (avec profondeur Z)."""
    
    # 1. Collecte des coordonnées X/Y/Z
    features = []
    for i in range(33):
        features.extend([landmarks[i].x, landmarks[i].y, landmarks[i].z])
    features = np.array(features)

    # 2. Calcul de la profondeur moyenne
    z_values = [landmarks[i].z for i in range(33)]
    mean_z = np.mean(z_values)
    if mean_z == 0:
        mean_z = 1.0
    
    # 3. Normalisation par Translation (Centrage sur hanche droite)
    ref_x = landmarks[RIGHT_HIP_LM].x
    ref_y = landmarks[RIGHT_HIP_LM].y
    
    normalized_features = np.zeros_like(features)
    for i in range(33):
        # X, Y normalisés par profondeur
        normalized_features[i*3] = (features[i*3] - ref_x) / mean_z
        normalized_features[i*3+1] = (features[i*3+1] - ref_y) / mean_z
        # Z non normalisé
        normalized_features[i*3+2] = features[i*3+2]

    # 4. Normalisation par Scaling (Échelle par distance inter-hanche)
    hip_dist = np.sqrt(
        (landmarks[RIGHT_HIP_LM].x - landmarks[LEFT_HIP_LM].x)**2 +
        (landmarks[RIGHT_HIP_LM].y - landmarks[LEFT_HIP_LM].y)**2
    )
    if hip_dist < 0.05:
        hip_dist = 1.0 
    
    for i in range(33):
        normalized_features[i*3] /= hip_dist
        normalized_features[i*3+1] /= hip_dist
    
    return normalized_features.tolist()


# --- Boucle de Détection en Temps Réel ---
def start_real_time_detection():
    
    VIDEO_FILE_PATH = BASE_DIR / 'Videos' / 'Videos' / 'PDFE01_1.mp4'
    cap = cv2.VideoCapture(str(VIDEO_FILE_PATH)) 
    
    if not cap.isOpened():
        print(f"❌ Erreur: Impossible d'ouvrir {VIDEO_FILE_PATH}. Vérifiez le chemin.")
        return

    print(f"📹 Détection sur vidéo: {VIDEO_FILE_PATH}")
    print(f"🎯 Seuil FoG: {THRESHOLD}")
    print("\n🔴 Appuyez sur ESC pour arrêter\n")
    
    fog_count = 0
    frame_count = 0
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print(f"\n✅ Fin de la vidéo atteinte après {frame_count} frames")
                break 

            frame_count += 1
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            fog_probability = 0.0
            fog_detected = False
            
            if results.pose_landmarks:
                # Normaliser et ajouter au buffer
                normalized_features = normalize_features(results.pose_landmarks.landmark)
                pose_buffer.append(normalized_features)
                
                # Prédiction si buffer plein
                if len(pose_buffer) == WINDOW_SIZE:
                    input_sequence = np.array(list(pose_buffer), dtype=np.float32).reshape(1, WINDOW_SIZE, N_FEATURES)
                    fog_probability = FOG_MODEL.predict(input_sequence, verbose=0)[0][0]
                    fog_detected = fog_probability >= THRESHOLD
                    
                    if fog_detected:
                        fog_count += 1
            
            # Affichage
            h, w, c = image.shape
            
            # Dessiner landmarks
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )
            
            # Afficher la probabilité FoG
            text = f"FoG: {fog_probability:.2%}" if len(pose_buffer) == WINDOW_SIZE else "FoG: Chargement..."
            color = (0, 0, 255) if fog_detected else (0, 255, 0)
            cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Afficher infos
            cv2.putText(image, f"Frame: {frame_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"FoG Events: {fog_count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('FoG Detection', image)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print(f"\n[STOP] Arret par l'utilisateur")
                break 

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n[STATS] Resume:")
    print(f"   Frames traitees: {frame_count}")
    print(f"   Evenements FoG: {fog_count}")
    if frame_count > 0:
        print(f"   Taux FoG: {fog_count/frame_count*100:.1f}%")


if __name__ == "__main__":
    start_real_time_detection()