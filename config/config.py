# config.py
from pathlib import Path
import mlflow

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR,"data")

#create dirs
DATA_DIR.mkdir(parents=True,exist_ok=True)

# Assets
DUPLICATE_QUESTIONS_URL = "https://raw.githubusercontent.com/Vishan007/Duplicate-question-mlops-end-to-end/main/data/sample_adv_features.csv"

STORES_DIR = Path(BASE_DIR, "stores")
MODEL_REGISTRY = Path(STORES_DIR, "model")
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri("file:///" + str(MODEL_REGISTRY.absolute()))

##to store our data assets -DVC for versioning our artifacts
BLOB_STORE = Path(STORES_DIR, "blob")
BLOB_STORE.mkdir(parents=True, exist_ok=True)
