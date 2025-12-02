import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration pour MediaPipe ---
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_FOLDER = BASE_DIR / 'data' / 'features'
OUTPUT_FOLDER = BASE_DIR / 'data' / 'preprocessed'
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Landmarks MediaPipe (33 points)
RIGHT_HIP_LM = 24  # Hanche droite
LEFT_HIP_LM = 23   # Hanche gauche
X_COL = f'lm_{RIGHT_HIP_LM}_x'
Y_COL = f'lm_{RIGHT_HIP_LM}_y'
Z_COL = f'lm_{RIGHT_HIP_LM}_z'
WINDOW_SIZE = 50

def normalize_and_segment(df_video, window_size=WINDOW_SIZE):
    """
    Normalise les coordonnées MediaPipe en tenant compte de la profondeur Z (distance caméra).
    - Centrage sur la hanche droite
    - Normalisation par profondeur moyenne
    - Normalisation par distance entre hanches
    Crée des séquences temporelles.
    """
    df_normalized = df_video.copy()
    
    # 1. Calcul de la profondeur moyenne (pour normaliser la distance à la caméra)
    z_cols = [f'lm_{i}_z' for i in range(33)]
    mean_z = df_normalized[z_cols].mean(axis=1)
    mean_z[mean_z == 0] = 1.0  # Évite division par zéro
    
    # 2. Centrage sur la hanche droite
    center_x = df_normalized[X_COL].values
    center_y = df_normalized[Y_COL].values
    
    # 3. Normalisation XY en tenant compte de la profondeur Z
    # (compense la différence de taille due à la distance à la caméra)
    for i in range(33):
        # Centrer
        df_normalized.loc[:, f'lm_{i}_x'] = df_normalized[f'lm_{i}_x'] - center_x
        df_normalized.loc[:, f'lm_{i}_y'] = df_normalized[f'lm_{i}_y'] - center_y
        
        # Normaliser par profondeur
        df_normalized.loc[:, f'lm_{i}_x'] = df_normalized[f'lm_{i}_x'] / mean_z
        df_normalized.loc[:, f'lm_{i}_y'] = df_normalized[f'lm_{i}_y'] / mean_z
    
    # 4. Étape de Mise à l'Échelle (Scaling par distance hanches)
    # Calcule la distance entre les hanches (droite et gauche) comme facteur d'échelle supplémentaire
    hip_distance = np.sqrt(
        (df_normalized[f'lm_{RIGHT_HIP_LM}_x'] - df_normalized[f'lm_{LEFT_HIP_LM}_x'])**2 +
        (df_normalized[f'lm_{RIGHT_HIP_LM}_y'] - df_normalized[f'lm_{LEFT_HIP_LM}_y'])**2
    )
    hip_distance[hip_distance == 0] = 1.0 
    
    # Normaliser par distance des hanches
    for i in range(33):
        df_normalized.loc[:, f'lm_{i}_x'] /= hip_distance
        df_normalized.loc[:, f'lm_{i}_y'] /= hip_distance
    
    # --- 5. Création des Séquences Temporelles ---
    # Utiliser X, Y normalisés + Z original (pour garder info profondeur)
    feature_cols = [col for col in df_normalized.columns if col.endswith('_x') or col.endswith('_y') or col.endswith('_z')]
    
    sequences = []
    
    for i in range(len(df_normalized) - window_size + 1):
        # Fenêtre temporelle
        sequence = df_normalized.iloc[i : i + window_size][feature_cols].values
        sequences.append(sequence)
        
    return np.array(sequences) if sequences else None


def preprocess_all_videos(features_folder=FEATURES_FOLDER):
    """
    Prétraite tous les fichiers CSV d'extraction MediaPipe du dossier features.
    Normalise en tenant compte de la profondeur Z.
    """
    csv_files = sorted(features_folder.glob('*.csv'))
    
    if not csv_files:
        print(f"❌ Aucun fichier CSV trouvé dans {features_folder}")
        return None
    
    print(f"📂 Traitement de {len(csv_files)} fichiers CSV...")
    print("=" * 70)
    
    all_sequences = []
    video_names = []
    
    for idx, csv_path in enumerate(csv_files, 1):
        try:
            print(f"[{idx:2d}/{len(csv_files)}] {csv_path.name:30s} → ", end="", flush=True)
            
            # Charger le CSV
            df = pd.read_csv(csv_path)
            
            if len(df) < WINDOW_SIZE:
                print(f"⚠️  {len(df)} frames < {WINDOW_SIZE}")
                continue
            
            # Prétraiter avec normalisation profondeur
            sequences = normalize_and_segment(df, WINDOW_SIZE)
            
            if sequences is not None:
                all_sequences.append(sequences)
                video_names.append(csv_path.stem)
                print(f"✓ {sequences.shape[0]:4d} séquences")
            else:
                print("✗ Pas de séquences")
                
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
    
    if not all_sequences:
        print("❌ Aucune séquence valide générée!")
        return None
    
    # Concaténer tous les fichiers
    X_all = np.concatenate(all_sequences, axis=0)
    
    print("=" * 70)
    print(f"✅ Prétraitement terminé!")
    print(f"   Vidéos traitées: {len(video_names)}/{len(csv_files)}")
    print(f"   Dimensions: ({X_all.shape[0]} séquences, {X_all.shape[1]} frames, {X_all.shape[2]} features)")
    
    # Sauvegarder les données
    output_path = OUTPUT_FOLDER / 'X_sequences.npy'
    np.save(output_path, X_all)
    print(f"\n💾 Données: {output_path}")
    
    # Sauvegarder les noms de vidéos
    video_names_path = OUTPUT_FOLDER / 'video_names.txt'
    with open(video_names_path, 'w') as f:
        for name in video_names:
            f.write(name + '\n')
    print(f"💾 Noms: {video_names_path}")
    
    return X_all


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PRÉTRAITEMENT DES DONNÉES MEDIAPIPE")
    print("Normalisation avec profondeur Z (distance à la caméra)")
    print("=" * 70 + "\n")
    
    X_sequences = preprocess_all_videos()
    
    if X_sequences is not None:
        print("\n✅ Prêt pour l'entraînement du modèle!")