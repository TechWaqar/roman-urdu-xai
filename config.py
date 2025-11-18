import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
BATCH_SIZE = 32
MAX_LENGTH = 128

# Create directories if they don't exist
for path in [DATA_PROCESSED, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(path, exist_ok=True)