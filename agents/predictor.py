from abc import ABC, abstractmethod
import numpy as np

class Predictor(ABC):
    """
    Abstract Base Class for Model Inference.
    Enforces the Dependency Inversion Principle.
    """
    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Runs inference on a preprocessed image array.
        
        Args:
            image (np.ndarray): Preprocessed image.
            
        Returns:
            np.ndarray: Prediction scores (e.g. probabilities).
        """
        pass
