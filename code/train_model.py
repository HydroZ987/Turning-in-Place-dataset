import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force GPU 0

import pandas as pd
import numpy as np
from pathlib import Path
import re
import tensorflow as tf

# Verifier GPU disponible
print("[GPU CHECK] Devices:", tf.config.list_physical_devices())

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
PREPROCESSED_FOLDER = BASE_DIR / 'data' / 'preprocessed'
ANNOTATIONS_FILE = BASE_DIR / 'PDFEinfo.csv'
MODEL_PATH = BASE_DIR / 'code' / 'best_fog_detector.keras'

WINDOW_SIZE = 50
THRESHOLD_FPS = 30  # Suppose pour les videos

print("=" * 70)
print("ENTRAÎNEMENT DU MODÈLE LSTM POUR LA DÉTECTION FoG")
print("=" * 70)

# ============================================================================
# 1. CHARGER LES DONNEES PRETRAITEES
# ============================================================================
print("\n[1] Chargement des donnees pretraitees...")
X_sequences = np.load(PREPROCESSED_FOLDER / 'X_sequences.npy')
print(f"[OK] Donnees chargees: {X_sequences.shape}")

# Lire les noms des videos
with open(PREPROCESSED_FOLDER / 'video_names.txt', 'r') as f:
    video_names = [line.strip() for line in f.readlines()]
print(f"[OK] {len(video_names)} videos trouvees")

# ============================================================================
# 2. CHARGER LES ANNOTATIONS FoG
# ============================================================================
print("\n[2] Chargement des annotations FoG...")
# Essayer differents encodages
encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-8']
df_annotations = None
for enc in encodings:
    try:
        df_annotations = pd.read_csv(ANNOTATIONS_FILE, sep=';', encoding=enc)
        print(f"[OK] CSV charge avec encodage: {enc}")
        break
    except Exception as e:
        continue

if df_annotations is None:
    print("[ERR] Impossible de charger le CSV avec les encodages connus")
    exit()

print(f"[OK] {len(df_annotations)} patients trouves")

def parse_fog_intervals(fog_string):
    """Parse les intervalles de FoG comme '[1.383-35.768; 36.696-65.969; ...]'"""
    if pd.isna(fog_string) or fog_string == '' or fog_string == '0':
        return []
    
    try:
        # Nettoyer la chaine
        fog_string = str(fog_string).strip('[]')
        if not fog_string or fog_string == '0' or fog_string == '0.00':
            return []
        
        intervals = []
        # Trouver tous les intervalles [start-end]
        matches = re.findall(r'(\d+\.?\d*)-(\d+\.?\d*)', fog_string)
        for start, end in matches:
            intervals.append((float(start), float(end)))
        return intervals
    except:
        return []

def get_fog_labels_for_video(video_name, fps=THRESHOLD_FPS):
    """Cree les labels FoG frame-par-frame pour une video."""
    # Extraire l'ID du patient (ex: PDFE01_1 -> PDFE01)
    patient_id = video_name.split('_')[0]
    
    # Chercher le patient dans les annotations
    patient_rows = df_annotations[df_annotations['ID'].str.strip() == patient_id]
    
    if len(patient_rows) == 0:
        print(f"[WARN] Patient {patient_id} non trouve dans les annotations")
        return None
    
    patient = patient_rows.iloc[0]
    
    # Determiner quelle session (base sur le numero de video)
    session_num = int(video_name.split('_')[-1]) if '_' in video_name else 1
    session_num = min(session_num, 3)  # Max 3 sessions
    
    # Colonnes pour cette session
    fog_time_col = f'Session {session_num} - time of FoG (s)'
    total_fog_col = f'Session {session_num} - total time in FoG (s)'
    
    # Recuperer les intervalles de FoG
    fog_intervals = parse_fog_intervals(patient[fog_time_col])
    total_fog_time = patient[total_fog_col]
    
    # Estimer la duree video (rough estimate)
    video_duration = 120  # Assume 120 secondes par defaut
    
    # Creer les labels: 1 si dans FoG, 0 sinon
    n_frames = int(video_duration * fps)
    labels = np.zeros(n_frames, dtype=int)
    
    # Marquer les frames en FoG
    for start, end in fog_intervals:
        start_frame = int(start * fps)
        end_frame = int(end * fps)
        labels[start_frame:end_frame] = 1
    
    return labels, len(fog_intervals), total_fog_time

# ============================================================================
# 3. CREER LES LABELS POUR TOUTES LES SEQUENCES
# ============================================================================
print("\n[3] Creation des labels FoG...")
y_labels = []
sequences_per_video = []

for video_name in video_names:
    result = get_fog_labels_for_video(video_name)
    if result is None:
        continue
    
    labels, n_fog_episodes, total_fog_time = result
    
    # Chaque sequence prend le label du dernier frame
    # (ou on peut prendre la majorite, a adapter selon vos besoins)
    for i in range(len(labels) - WINDOW_SIZE + 1):
        # Label = 1 si au moins 1 frame du window est en FoG
        window_label = 1 if np.any(labels[i:i+WINDOW_SIZE] == 1) else 0
        y_labels.append(window_label)

y_labels = np.array(y_labels)

print(f"[OK] {len(y_labels)} labels crees")
print(f"     FoG: {np.sum(y_labels)} sequences ({np.sum(y_labels)/len(y_labels)*100:.1f}%)")
print(f"     Non-FoG: {len(y_labels) - np.sum(y_labels)} sequences ({(1-np.sum(y_labels)/len(y_labels))*100:.1f}%)")

# Verifier que les tailles correspondent
if len(y_labels) != len(X_sequences):
    print(f"[WARN] Attention: {len(X_sequences)} sequences vs {len(y_labels)} labels")
    # Adapter si necessaire
    min_len = min(len(X_sequences), len(y_labels))
    X_sequences = X_sequences[:min_len]
    y_labels = y_labels[:min_len]

# ============================================================================
# 4. SEPARATION TRAIN/TEST
# ============================================================================
print("\n[4] Separation train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, 
    y_labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_labels
)

print(f"[OK] Train: {X_train.shape[0]} sequences")
print(f"[OK] Test: {X_test.shape[0]} sequences")

# ============================================================================
# 5. GESTION DU DESEQUILIBRE DE CLASSES
# ============================================================================
print("\n[5] Calcul des poids de classes...")
class_weights = class_weight.compute_class_weight(
    'balanced', 
    classes=np.unique(y_train), 
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))
print(f"[OK] Poids: {class_weights_dict}")

# ============================================================================
# 6. CONSTRUCTION DU MODELE LSTM
# ============================================================================
print("\n[6] Construction du modele LSTM...")
model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(WINDOW_SIZE, X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=64, return_sequences=False),
    Dropout(0.2),
    Dense(units=32, activation='relu'),
    Dropout(0.1),
    Dense(units=1, activation='sigmoid')  # Classification binaire
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'Recall', 'Precision']
)

print(model.summary())

# ============================================================================
# 7. ENTRAÎNEMENT
# ============================================================================
print("\n[7] Demarrage de l'entraînement...")
print(f"     Epochs: 20 (reduit pour utiliser GPU efficacement)")
print(f"     Batch size: 128 (augmente pour GPU)")
print(f"     Validation split: 0.2")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# 8. EVALUATION
# ============================================================================
print("\n[8] Evaluation du modele...")
test_loss, test_acc, test_recall, test_precision = model.evaluate(
    X_test, y_test, 
    verbose=0
)

print(f"[OK] Resultats sur le set de test:")
print(f"     Loss: {test_loss:.4f}")
print(f"     Accuracy: {test_acc:.4f}")
print(f"     Recall: {test_recall:.4f}")
print(f"     Precision: {test_precision:.4f}")

# Calculer F1-score
f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-10)
print(f"     F1-Score: {f1:.4f}")

print(f"\n[OK] Modele sauvegarde: {MODEL_PATH}")
print("=" * 70)
