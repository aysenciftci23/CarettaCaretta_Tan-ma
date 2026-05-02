import os
import json
import numpy as np
from config import ID_MAP_PATH

class MatchingAgent:
    """
    Agent responsible for translating probability scores into Turtle IDs.
    Single Responsibility: ID Mapping.
    """
    def __init__(self):
        self.id_to_label = {}
        self.label_to_id = {}
        self._load_map()

    def _load_map(self):
        if not os.path.exists(ID_MAP_PATH):
            print(f"[MatchingAgent] WARNING: {ID_MAP_PATH} not found. Creating mock mapping.")
            # Create a mock mapping for 438 classes if it doesn't exist
            self.label_to_id = {i: f"MOCK_ID_{i}" for i in range(438)}
            # Also save the mock to the file for the user to inspect/replace
            os.makedirs(os.path.dirname(ID_MAP_PATH), exist_ok=True)
            with open(ID_MAP_PATH, 'w') as f:
                json.dump({f"MOCK_ID_{i}": i for i in range(438)}, f, indent=4)
        else:
            try:
                with open(ID_MAP_PATH, 'r') as f:
                    self.id_to_label = json.load(f)
                # Map integer index back to Turtle ID
                self.label_to_id = {int(v): str(k) for k, v in self.id_to_label.items()}
                print("[MatchingAgent] ID mapping loaded successfully.")
            except Exception as e:
                print(f"[MatchingAgent] Failed to load ID map: {e}")
                raise

    def match(self, probabilities: np.ndarray) -> dict:
        """
        Finds the highest scoring class and returns the Turtle ID.
        
        Args:
            probabilities (np.ndarray): 1D array of scores for each class.
            
        Returns:
            dict: Contains 'turtle_id' and 'confidence'.
        """
        try:
            best_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[best_idx])
            
            turtle_id = self.label_to_id.get(best_idx, f"UNKNOWN_CLASS_{best_idx}")
            
            return {
                "turtle_id": turtle_id,
                "confidence": confidence
            }
        except Exception as e:
            print(f"[MatchingAgent] Matching error: {e}")
            raise
