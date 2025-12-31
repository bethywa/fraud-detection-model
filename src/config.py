from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# Model path
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"

# Random seed
RANDOM_STATE = 42

# SHAP
SHAP_SAMPLE_SIZE = 2000
