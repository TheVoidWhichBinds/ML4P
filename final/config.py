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

NUM_PHYSICAL_CHANNELS = 4 # always true for this dataset
SPATIAL_KERNEL_SIZE = 3 # 3x3 local patch

# The WELL domain wraps in x and copies the edge in y.
X_BOUNDARY_MODE = "periodic"
Y_BOUNDARY_MODE = "replicate" 

K_VALUES = [25, 50, 75, 100] # histories to test
SLOPE_K = min(K_VALUES) # k used for the slope penalty








#================================================================================================================================================
# INNER MODEL
#================================================================================================================================================

TAYLOR_ALPHA = 10 # number of Taylor terms
TAYLOR_X0 = 0.0
TAYLOR_CLAMP_VALUE = 10.0

HIDDEN_CHANNELS = 32
TEMPORAL_KERNEL_SIZE = 3
NUM_CNN_LAYERS = 3








#================================================================================================================================================
# TRAINING
#================================================================================================================================================

BATCH_SIZE = 50
NUM_EPOCHS = 6
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP_MAX_NORM = 1.0
VRMSE_EPS = 1.0e-7
NUM_WORKERS = 2
PIN_MEMORY = False


EPOCH_PROGRESS_DIVISIONS = 4 

PREDICTION_WEIGHT = 1.0

slope_loss = True # adds the exponential slope penalty
SLOPE_WEIGHT = 1.0e-1
SLOPE_FIT_STEPS = 25
SLOPE_FIT_LR = 5.0e-2

MAX_FILES = None
MAX_TRAJECTORIES = 1
MAX_SAMPLES = 100000 # training sample cap

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


N_EPOCHS = 4 # saved epochs to plot
MAX_PARITY_SAMPLES = 10000 # max parity samples saved per k
PARITY_PLOT_MAX_POINTS = 50000 # max parity points drawn








#================================================================================================================================================
# QUICK TEST SETTINGS
#================================================================================================================================================

TEST_RUN_NUM_EPOCHS = 1
TEST_RUN_MAX_FILES = 1
TEST_RUN_MAX_TRAJECTORIES = 1
TEST_RUN_MAX_SAMPLES = 64
TEST_RUN_MAX_BATCHES = 1
