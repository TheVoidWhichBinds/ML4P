# config.py

from pathlib import Path








#================================================================================================================================================
# DATA
#================================================================================================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "turbulent_radiative_layer_2D" / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

NUM_PHYSICAL_CHANNELS = 4
SPATIAL_KERNEL_SIZE = 3

# The turbulent_radiative_layer_2D domain is periodic in x and zero-gradient in y.
# For tensors shaped [T, C, H, W], this means W wraps and H uses replicate padding.
X_BOUNDARY_MODE = "periodic"
Y_BOUNDARY_MODE = "replicate"

K_VALUES = [10, 25, 50, 75, 100]
SLOPE_K = 10








#================================================================================================================================================
# INNER MODEL
#================================================================================================================================================

TAYLOR_ALPHA = 3
TAYLOR_X0 = 0.0

HIDDEN_CHANNELS = 32
TEMPORAL_KERNEL_SIZE = 3
MLP_HIDDEN_DIMS = [128, 64, 32]








#================================================================================================================================================
# TRAINING
#================================================================================================================================================

BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1.0e-7
WEIGHT_DECAY = 0.0
GRAD_CLIP_MAX_NORM = 1.0
NUM_WORKERS = 0
PIN_MEMORY = False

# Number of within-epoch progress checkpoints to print for each train/validation pass.
# For example, 4 prints around 25%, 50%, 75%, and 100%.
EPOCH_PROGRESS_DIVISIONS = 0

PREDICTION_WEIGHT = 1.0
EXPONENTIAL_FIT_WEIGHT = 1.0e-1
SLOPE_WEIGHT = 0.0

MAX_FILES = None
MAX_TRAJECTORIES = 1
MAX_SAMPLES = 64

TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"

RANDOM_SEED = 1234








#================================================================================================================================================
# QUICK TEST SETTINGS
#================================================================================================================================================

TEST_RUN_NUM_EPOCHS = 1
TEST_RUN_MAX_FILES = 1
TEST_RUN_MAX_TRAJECTORIES = 1
TEST_RUN_MAX_SAMPLES = 64
TEST_RUN_MAX_BATCHES = 1
