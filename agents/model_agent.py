import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

from .predictor import Predictor
from config import MODEL_PATH

class ModelAgent(Predictor):
    """
    Keras Model Agent.
    Loads a pretrained Keras model and implements the Predictor interface.
    """
    def __init__(self):
        print(f"[ModelAgent] Loading model from {MODEL_PATH}...")
        try:
            self.model = load_model(MODEL_PATH)
            print("[ModelAgent] Model loaded successfully.")
        except OSError as e:
            print(f"[ModelAgent] Failed to load model: {e}")
            raise

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Runs the image through the EfficientNetB3 preprocessing and model.
        
        Args:
            image (np.ndarray): Image array of shape (224, 224, 3)
            
        Returns:
            np.ndarray: Probability scores for each class.
        """
        try:
            # Check shape
            if image.shape != (224, 224, 3):
                raise ValueError(f"Expected image shape (224, 224, 3), got {image.shape}")
                
            # Expand dimensions to create a batch of 1
            image_batch = np.expand_dims(image, axis=0) # (1, 224, 224, 3)
            
            # Apply EfficientNet preprocessing
            preprocessed_image = preprocess_input(image_batch)
            
            # Predict
            predictions = self.model.predict(preprocessed_image, verbose=0)
            
            return predictions[0] # Return the 1D array of scores for the single image
            
        except Exception as e:
            print(f"[ModelAgent] Prediction error: {e}")
            raise
