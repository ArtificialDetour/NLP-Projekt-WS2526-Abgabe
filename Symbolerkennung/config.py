import os
from pathlib import Path

# Projektverzeichnis relativ zu dieser Datei
BASE_DIR = Path(__file__).resolve().parent

# Datenpfade
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
INPUT_DIR = BASE_DIR / "input"
INPUT_PARTS_DIR = INPUT_DIR / "parts"
INPUT_DIAGRAM_DIR = INPUT_DIR / "diagram"

os.makedirs(INPUT_PARTS_DIR, exist_ok=True)
os.makedirs(INPUT_DIAGRAM_DIR, exist_ok=True)

# Ausgabepfade
OUTPUT_DIR = BASE_DIR / "output"
WEIGHTS_DIR = OUTPUT_DIR / "weights"

os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Modell-Konfiguration: vortrainierter ViT-Base mit 16×16-Patches
MODEL_NAME = "google/vit-base-patch16-224"
NUM_CLASSES = 5
CLASSES = ["action", "choice", "ending", "start", "state"]

# Trainingsparameter
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Bildgröße (ViT erwartet 224×224)
IMAGE_SIZE = 224
