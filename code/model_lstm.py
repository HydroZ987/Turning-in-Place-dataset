import cv2
import mediapipe as mp
import numpy as np
import collections
from tensorflow.keras.models import load_model
from pathlib import Path

# --- CONSTANTES IDENTIQUES À L'ENTRAÎNEMENT ET AU PRÉTRAITEMENT ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'best_fog_detector.keras'

# ⚠️ PARAMÈTRES TIRÉS DE VOS SCRIPTS
WINDOW_SIZE = 50 
N_LANDMARKS = 33
N_FEATURES = N_LANDMARKS * 2  # 66 features (X et Y pour 33 landmarks)
THRESHOLD = 0.8  # Seuil de probabilité pour l'alerte FoG (ajustable)
RIGHT_HIP_LM = 24 # Landmark de la hanche droite (pour le centrage)
LEFT_HIP_LM = 23  # Landmark de la hanche gauche (pour l'échelle)

# --- Initialisation ---
try:
    # On charge le modèle une seule fois au démarrage
    FOG_MODEL = load_model(MODEL_PATH)
    print("✅ Modèle FoG chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur de chargement du modèle: {e}. Vérifiez que 'best_fog_detector.keras' est dans le dossier 'models'.")
    exit()

mp_pose = mp.solutions.pose
# collections.deque est utilisé pour maintenir un buffer glissant de taille fixe
pose_buffer = collections.deque(maxlen=WINDOW_SIZE) 

def normalize_features(landmarks):
    """
    Applique la même logique de normalisation (Centrage + Scaling)
    que celle utilisée dans src/02_preprocess_data.py.
    """
    
    # 1. Collecte des coordonnées X/Y brutes
    # Le landmark 24 (Hanche droite) est la référence pour la translation
    ref_x = landmarks[RIGHT_HIP_LM].x
    ref_y = landmarks[RIGHT_HIP_LM].y
    
    current_features = []
    for i in range(N_LANDMARKS):
        # 2. Normalisation par Translation (Centrage)
        # On soustrait la référence à toutes les coordonnées
        current_features.append(landmarks[i].x - ref_x) 
        current_features.append(landmarks[i].y - ref_y)
    
    normalized_features = np.array(current_features)

    # 3. Normalisation par Scaling (Mise à l'échelle)
    # Distance entre les hanches (24 et 23) pour l'échelle
    hip_dist = np.sqrt(
        (landmarks[RIGHT_HIP_LM].x - landmarks[LEFT_HIP_LM].x)**2 +
        (landmarks[RIGHT_HIP_LM].y - landmarks[LEFT_HIP_LM].y)**2
    )
    
    # On utilise 1.0 comme valeur plancher pour éviter la division par zéro 
    # et gérer les détections incertaines.
    if hip_dist < 0.05: 
        hip_dist = 1.0 
    
    normalized_features /= hip_dist
    
    return normalized_features.tolist() # Retourne la liste de 66 features normalisées

# --- Boucle de Détection en Temps Réel ---
def start_real_time_detection():
    # 0 pour la webcam par défaut, ou chemin vers un fichier vidéo pour le test
    cap = cv2.VideoCapture(0) 
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        print("Démarrage de la détection...")
        while cap.isOpened():
            success, image = cap.read()
            if not success: continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            fog_probability = 0.0
            
            if results.pose_landmarks:
                
                # 1. Obtenir les features normalisées (CRUCIAL : utilise la fonction ci-dessus)
                normalized_features = normalize_features(results.pose_landmarks.landmark)
                
                # 2. Mettre à jour le buffer (collections.deque s'occupe de la taille)
                pose_buffer.append(normalized_features)
                
                # 3. Prédiction (seulement quand le buffer est plein, soit 50 trames)
                if len(pose_buffer) == WINDOW_SIZE:
                    # Préparer les données pour le modèle: (1, 50, 66)
                    input_sequence = np.array(pose_buffer, dtype=np.float32).reshape(1, WINDOW_SIZE, N_FEATURES)
                    
                    # Faire la prédiction
                    # verbose=0 pour ne pas afficher la progression
                    fog_probability = FOG_MODEL.predict(input_sequence, verbose=0)[0][0]
                    
                    # 4. Affichage de l'alerte
                    if fog_probability >= THRESHOLD:
                        alerte_text = f"!!! ALERTE FOG: {fog_probability:.2f} !!!"
                        couleur = (0, 0, 255) # Rouge
                        # Ajoutez ici la logique de déclenchement d'une alarme (son, notification)
                    else:
                        alerte_text = f"FoG Score: {fog_probability:.2f}"
                        couleur = (0, 255, 0) # Vert
                        
                    cv2.putText(image, alerte_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, couleur, 2)
            
            # Affichage de la pose par MediaPipe
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            cv2.imshow('FoG Detection', image)
            # Quitter avec la touche ESC
            if cv2.waitKey(5) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_real_time_detection()