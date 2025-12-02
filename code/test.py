import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# ⚠️ Assurez-vous que le chemin est correct
MODEL_PATH = 'best_fog_detector.keras'
FOG_MODEL = load_model(MODEL_PATH)