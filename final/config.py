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

TRAIN_CHECKPOINT_PATH = CHECKPOINT_DIR / "inner_shared.pt"
TRAIN_LOG_PATH = LOG_DIR / "inner_train_log.csv"
TEST_LOG_PATH = LOG_DIR / "inner_test_log.csv"
TEST_PARITY_PATH = LOG_DIR / "inner_test_parity.csv"

NUM_PHYSICAL_CHANNELS = 4
SPATIAL_KERNEL_SIZE = 3

# The turbulent_radiative_layer_2D domain is periodic in x and zero-gradient in y.
# For tensors shaped [T, C, H, W], this means W wraps and H uses replicate padding.
X_BOUNDARY_MODE = "periodic"
Y_BOUNDARY_MODE = "replicate"

# Temporal-history values for the full multi-k run.
# Edit this list directly to choose the k values used by run.py.
# Every k must be positive and smaller than the number of timestamps in each trajectory.
K_VALUES = [80, 85, 90, 95, 100]
SLOPE_K = min(K_VALUES)








#================================================================================================================================================
# INNER MODEL
#================================================================================================================================================

TAYLOR_ALPHA = 5
TAYLOR_X0 = 0.0
TAYLOR_CLAMP_VALUE = 10.0

HIDDEN_CHANNELS = 32
TEMPORAL_KERNEL_SIZE = 3
NUM_CNN_LAYERS = 3








#================================================================================================================================================
# TRAINING
#================================================================================================================================================

BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1.0e-7
WEIGHT_DECAY = 0.0
GRAD_CLIP_MAX_NORM = 1.0
VRMSE_EPS = 1.0e-7
NUM_WORKERS = 0
PIN_MEMORY = False

# Number of within-epoch progress checkpoints to print for each train/validation pass.
# For example, 4 prints around 25%, 50%, 75%, and 100%.
EPOCH_PROGRESS_DIVISIONS = 4

PREDICTION_WEIGHT = 1.0

# If True, run.py optimizes:
#
#     pooled VRMSE + SLOPE_WEIGHT * exponential_slope_loss
#
# The exponential parameters p, A, k_shift, and w are not trainable model parameters.
# They are refit from the current VRMSE(k) values inside each loss evaluation.
# If False, run.py optimizes pooled VRMSE only.
slope_loss = True
SLOPE_WEIGHT = 1.0e-1
SLOPE_FIT_STEPS = 25
SLOPE_FIT_LR = 5.0e-2

MAX_FILES = None
MAX_TRAJECTORIES = 1
MAX_SAMPLES = 10000

TRAIN_SPLIT = "train"
VALID_SPLIT = "valid"
TEST_SPLIT = "test"

RANDOM_SEED = 1234








#================================================================================================================================================
# PRETRAINING / WARM START
#================================================================================================================================================

PRETRAIN_K = max(K_VALUES)
PRETRAIN_NUM_EPOCHS = NUM_EPOCHS
LOAD_PRETRAINED_INNERK = True
PRETRAIN_CHECKPOINT_PATH = CHECKPOINT_DIR / f"inner_k{PRETRAIN_K}_pretrain.pt"
PRETRAIN_TEST_CHECKPOINT_PATH = CHECKPOINT_DIR / f"inner_k{PRETRAIN_K}_pretrain_test.pt"
PRETRAIN_LOG_PATH = LOG_DIR / f"inner_k{PRETRAIN_K}_pretrain_log.csv"
PRETRAIN_TEST_LOG_PATH = LOG_DIR / f"inner_k{PRETRAIN_K}_pretrain_test_log.csv"









#================================================================================================================================================
# PLOTTING
#================================================================================================================================================

# Number of evenly spaced saved epochs to plot in plot.py.
# The final saved epoch is always included.
N_EPOCHS = 4

# Maximum number of test-set samples per k saved by test.py for parity plotting.
# Each saved sample produces one parity row per physical channel.
MAX_PARITY_SAMPLES = 10000

# Maximum number of points drawn on plot.py parity plots after loading the parity CSV.
PARITY_PLOT_MAX_POINTS = 50000








#================================================================================================================================================
# QUICK TEST SETTINGS
#================================================================================================================================================

TEST_RUN_NUM_EPOCHS = 1
TEST_RUN_MAX_FILES = 1
TEST_RUN_MAX_TRAJECTORIES = 1
TEST_RUN_MAX_SAMPLES = 64
TEST_RUN_MAX_BATCHES = 1
