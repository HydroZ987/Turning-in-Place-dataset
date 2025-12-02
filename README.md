# Turning-in-Place Dataset

Ce dépôt contient le code et les données pour la détection du freezing of gait (FOG) chez les personnes atteintes de la maladie de Parkinson lors de la tâche de rotation sur place.

## 📂 Structure du projet

```
Turning-in-Place-dataset/
├── code/                           # Scripts Python pour l'analyse
│   ├── train_model.py             # Entraînement du modèle LSTM
│   ├── test.py                    # Test du modèle
│   ├── batch_test_fog.py          # Test par lot
│   ├── test_single_video.py       # Test sur une vidéo unique
│   ├── real_time_detection_by_cam.py      # Détection en temps réel par caméra
│   ├── real_time_detection_by_video.py    # Détection en temps réel par vidéo
│   ├── webcam_fog_detection.py    # Détection FOG par webcam
│   ├── extrac_data_videos.py      # Extraction des données vidéo
│   ├── pretraitement.py           # Prétraitement des données
│   ├── model_lstm.py              # Architecture du modèle LSTM
│   └── best_fog_detector.keras    # Modèle pré-entraîné
├── data/
│   ├── features/                  # Caractéristiques extraites (CSV)
│   └── preprocessed/              # Données prétraitées
├── IMU/                           # Données des capteurs inertiels
└── PDFEinfo.csv                   # Informations sur les participants
```

## 📥 Téléchargement des vidéos

**IMPORTANT** : Les fichiers vidéo ne sont pas inclus dans ce dépôt Git en raison de leur taille importante.

Vous devez télécharger le dossier `Videos/` depuis Figshare :

🔗 **[Télécharger les vidéos ici](https://figshare.com/articles/dataset/A_public_dataset_of_video_acceleration_and_angular_velocity_in_individuals_with_Parkinson_s_disease_during_the_turning-in-place_task/14984667)**

Une fois téléchargé, placez le dossier `Videos/Videos/` à la racine du projet :
```
Turning-in-Place-dataset/
├── Videos/
│   └── Videos/
│       ├── PDFE01_1.mp4
│       ├── PDFE01_2.mp4
│       └── ...
```

## 🛠️ Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Installation des dépendances

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn mediapipe
```

Ou créez un environnement virtuel :

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn mediapipe
```

## 🚀 Utilisation

### 1. Prétraitement des données

```bash
python code/pretraitement.py
```

Ce script génère le fichier `data/preprocessed/X_sequences.npy` nécessaire pour l'entraînement.

### 2. Entraînement du modèle

```bash
python code/train_model.py
```

### 3. Test du modèle

Test sur un ensemble de vidéos :
```bash
python code/test.py
```

Test sur une vidéo unique :
```bash
python code/test_single_video.py
```

### 4. Détection en temps réel

Avec une caméra :
```bash
python code/real_time_detection_by_cam.py
```

Avec un fichier vidéo :
```bash
python code/real_time_detection_by_video.py
```

## 📊 Dataset

Le dataset comprend :
- **35 participants** atteints de la maladie de Parkinson
- **Vidéos** : enregistrements de la tâche de rotation sur place (180° et 360°)
- **Données IMU** : accélération et vitesse angulaire synchronisées
- **Annotations** : instants de freezing of gait identifiés

### Format des données

- **Vidéos** : fichiers MP4 (disponibles sur Figshare)
- **Features** : fichiers CSV avec les caractéristiques extraites par frame
- **IMU** : fichiers CSV avec les mesures des capteurs inertiels

## 📝 Citation

Si vous utilisez ce dataset dans vos recherches, veuillez citer :

```
[Citation à ajouter depuis l'article Figshare]
```

## 📄 Licence

[À définir selon les termes du dataset Figshare]

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## ⚠️ Notes importantes

- Les fichiers `.npy`, `.zip` et `.mp4` sont exclus du dépôt Git (voir `.gitignore`)
- Assurez-vous d'avoir suffisamment d'espace disque pour les vidéos (~5 GB)
- Le modèle pré-entraîné `best_fog_detector.keras` est inclus dans le dossier `code/`

## 📧 Contact

Pour toute question, veuillez ouvrir une issue sur ce dépôt.
