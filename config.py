import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
AGENTS_DIR = os.path.join(BASE_DIR, 'agents')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Kaggle Dataset details
KAGGLE_DATASET = "wildlife-datasets/seaturtleid2022"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determine if we should use a subset (if CPU is used)
USE_SUBSET = not torch.cuda.is_available()
SUBSET_SIZE = 100  # Number of unique turtles to use if CPU

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0 # Windows'ta Dataloader hatalarını önlemek için 0 tutuldu
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# Keras Inference
MODEL_PATH = "sea_turtle_model_v1.keras"
CLASSES_COUNT = 438
ID_MAP_PATH = "models/id_to_label.json"
EMBEDDING_DIM = 2048

# Check results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
