from pathlib import Path
from the_well.data import WellDataset
from the_well.utils.download import well_download
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from ..TVT import TVT_MHD
from torch.utils.data import ConcatDataset, Subset
from .augment import Augmentation, b_parity, rotation_symmetry










#============= ADJUSTABLE HYPERPARAMETERS =====================================================================================
#================================================================
# Multiplication factor of train_aug_divisor should match number 
#   of concatenated data sets in DATA INITIALIZATION s.t. 
#   baseline NN and symmetry-enforcing NN both are given the same 
#   qty of data
TRAIN_ORIG_DIVISOR = 10
TRAIN_AUG_DIVISOR = 13 * TRAIN_ORIG_DIVISOR 
VALID_DIVISOR = 1
TEST_DIVISOR = 1

# Batch settings:
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
TEST_BATCH_SIZE = 2

# Model architecture settings:
SPATIAL_KERNEL_SIZE = 3
TEMPORAL_KERNEL_SIZE = 3
HIDDEN_CHANNELS = 4
ACTIVATION = nn.LeakyReLU()

# Additional model architecture settings:
INPUT_CHANNELS = 7
OUTPUT_CHANNELS = 7
SPATIAL_DOWN_STRIDE = 2
SPATIAL_LOWRES_STRIDE = 1
SPATIAL_UP_KERNEL_SIZE = 4
SPATIAL_UP_STRIDE = 2
SPATIAL_UP_PADDING = 1
TEMPORAL_STRIDE = 1
DEFAULT_ACTIVATION = nn.ReLU()

# Data settings:
DATASET_NAME = "MHD_64"
DATA_SPLITS = ["train", "valid", "test"]
N_STEPS_INPUT = 5
N_STEPS_OUTPUT = 1
USE_NORMALIZATION = False

# Optimizer settings:
LEARNING_RATE = 1e-3
LOSS_FUNC = nn.MSELoss()
EPOCHS = 10
DELTA_THRESHOLD = 1e-2

# DataLoader settings:
TRAIN_SHUFFLE = True
EVAL_SHUFFLE = False

# Plotting settings:
PLOTTING = True
#==================


#====================
#--------------------
# Both architectures:
architecture_config = {
    # DataLoader settings:
    "train_batch_size": TRAIN_BATCH_SIZE,
    "valid_batch_size": VALID_BATCH_SIZE,
    "test_batch_size": TEST_BATCH_SIZE,
    "train_shuffle": TRAIN_SHUFFLE,
    "eval_shuffle": EVAL_SHUFFLE,

    # Model architecture settings:
    "spatial_kernel_size": SPATIAL_KERNEL_SIZE,
    "temporal_kernel_size": TEMPORAL_KERNEL_SIZE,
    "hidden_channels": HIDDEN_CHANNELS,
    "activation": ACTIVATION,

    # Optimizer / training settings:
    "learning_rate": LEARNING_RATE,
    "loss_func": LOSS_FUNC,
    "epochs": EPOCHS,
    "delta_threshold": DELTA_THRESHOLD,
    "plotting": PLOTTING,
}
#--------------------

#----------------------
symmetry_MHD_config = {
    **architecture_config,
    "model_name": "Symmetry MHD",
}
baseline_MHD_config = {
    **architecture_config,
    "model_name": "Baseline MHD",
}
#--------------------------------
#================================
#=============================================================================================================================














#============= DATA INITIALIZATION ============================================================================================
#==================
# Downloading data:
project_dir = Path(__file__).resolve().parent.parent
base_path = project_dir / "datasets"

for split in DATA_SPLITS:
    split_dir = base_path / DATASET_NAME / "data" / split
    if not split_dir.exists():
        well_download(
            base_path = str(base_path),
            dataset = DATASET_NAME,
            split = split,
        )
#=========================


#========================================
# Splitting data into train, valid, test:
datasets = {} # initializing dictionary
for split in DATA_SPLITS:
    datasets[split] = WellDataset( # generating key-value pairs
        well_base_path = str(base_path),
        well_dataset_name = DATASET_NAME,
        well_split_name = split,
        n_steps_input = N_STEPS_INPUT,
        n_steps_output = N_STEPS_OUTPUT,
        use_normalization = USE_NORMALIZATION,
    )
#=================================


#=======================
def heldout_generator():
    """
    Generates shared validation and test subsets so that both
    architectures are evaluated on identical held-out data.
    """
    # Validation data:
    valid_orig = datasets["valid"]
    N_valid = len(valid_orig)
    valid_indices = torch.randperm(N_valid)[:(N_valid // VALID_DIVISOR)]
    valid_orig = Subset(valid_orig, valid_indices)

    # Test data:
    test_orig = datasets["test"]
    N_test = len(test_orig)
    test_indices = torch.randperm(N_test)[:(N_test // TEST_DIVISOR)]
    test_orig = Subset(test_orig, test_indices)

    print("Shared validation and test data prepared ...")
    return valid_orig, test_orig
#======================================================


#=======================
def train_generator():
    """
    Augments training data, and takes subsets of the train datasets.
    Validation and test are generated separately so they can be shared
    between both architectures.
    """
    # Data augmentation:
    train_orig = datasets["train"]
    train_mag = Augmentation(train_orig, transform=b_parity)
    train_z90 = Augmentation(
        train_orig,
        transform = lambda sample: rotation_symmetry(sample, axis_rot="z", num_90_rot=1)
    )
    train_z180 = Augmentation(
        train_orig,
        transform = lambda sample: rotation_symmetry(sample, axis_rot="z", num_90_rot=2)
    )
    train_z270 = Augmentation(
        train_orig,
        transform = lambda sample: rotation_symmetry(sample, axis_rot="z", num_90_rot=3)
    )
    train_x90 = Augmentation(
        train_orig,
        transform=lambda sample: rotation_symmetry(sample, axis_rot="x", num_90_rot=1)
    )
    train_x180 = Augmentation(
        train_orig,
        transform=lambda sample: rotation_symmetry(sample, axis_rot="x", num_90_rot=2)
    )
    train_x270 = Augmentation(
        train_orig,
        transform=lambda sample: rotation_symmetry(sample, axis_rot="x", num_90_rot=3)
    )
    train_y90 = Augmentation(
        train_orig,
        transform=lambda sample: rotation_symmetry(sample, axis_rot="y", num_90_rot=1)
    )
    train_y180 = Augmentation(
        train_orig,
        transform=lambda sample: rotation_symmetry(sample, axis_rot="y", num_90_rot=2)
    )
    train_y270 = Augmentation(
        train_orig,
        transform=lambda sample: rotation_symmetry(sample, axis_rot="y", num_90_rot=3)
    )
    #=================================================================================


    #===============================
    # Combining train augmentations:
    train_aug = ConcatDataset([
        train_orig, 
        train_mag, 
        train_mag, # B-field parity re-introduced for strength
        train_mag,
        train_z90, 
        train_z180,
        train_z270,
        train_x90,
        train_x180,
        train_x270, 
        train_y90,
        train_y180,
        train_y270
        ])
    N_train_aug = len(train_aug)
    indices = torch.randperm(N_train_aug)[:(N_train_aug // TRAIN_AUG_DIVISOR)]
    train_aug = Subset(train_aug, indices)

    # Train data for baseline:
    N_train_orig = len(train_orig)
    indices = torch.randperm(N_train_orig)[:(N_train_orig // TRAIN_ORIG_DIVISOR)]
    train_orig = Subset(train_orig, indices)

    print("Training data augmentation complete ...")
    return train_orig, train_aug
    #================================
#======================================================
#=============================================================================================================================










#============= MODEL ==========================================================================================
#==================================
class SpatioTemporalCNN(nn.Module):
    """
    Input:
        X : [N, T, X, Y, Z, 7]

    Output:
        Y_pred : [N, 1, X, Y, Z, 7]

    Architecture:
      1. Spatial encoder:
           full resolution -> lower resolution
      2. Temporal conv:
           operate over time at reduced spatial resolution
      3. Spatial decoder:
           lower resolution -> full resolution
    """

    #============
    def __init__(
        self,
        spatial_kernel_size: int = SPATIAL_KERNEL_SIZE,
        temporal_kernel_size: int = TEMPORAL_KERNEL_SIZE,
        hidden_channels: int = HIDDEN_CHANNELS,
        activation: nn.Module | None = None,
    ):
        super().__init__()

        if activation is None:
            activation = DEFAULT_ACTIVATION
        self.act = activation

        spatial_padding = spatial_kernel_size // 2
        temporal_padding = temporal_kernel_size // 2

        # -----------------------------
        # SPATIAL ENCODER
        # [N*T, 7, 64, 64, 64]
        # -> [N*T, hidden_channels, 32, 32, 32]
        # -----------------------------
        self.spatial_down = nn.Conv3d(
            in_channels = INPUT_CHANNELS,
            out_channels = hidden_channels,
            kernel_size = spatial_kernel_size,
            stride = SPATIAL_DOWN_STRIDE,
            padding = spatial_padding,
        )

        # Optional extra processing at low resolution
        self.spatial_lowres = nn.Conv3d(
            in_channels = hidden_channels,
            out_channels = hidden_channels,
            kernel_size = spatial_kernel_size,
            stride = SPATIAL_LOWRES_STRIDE,
            padding = spatial_padding,
        )

        # -----------------------------
        # TEMPORAL CNN
        # Input:
        #   [N*X_low*Y_low*Z_low, hidden_channels, T]
        # Output:
        #   same shape
        # -----------------------------
        self.temporal_conv = nn.Conv1d(
            in_channels = hidden_channels,
            out_channels = hidden_channels,
            kernel_size = temporal_kernel_size,
            stride = TEMPORAL_STRIDE,
            padding = temporal_padding,
        )

        # -----------------------------
        # SPATIAL DECODER
        # [N, hidden_channels, 32, 32, 32]
        # -> [N, 7, 64, 64, 64]
        # -----------------------------
        self.spatial_up = nn.ConvTranspose3d(
            in_channels = hidden_channels,
            out_channels = hidden_channels,
            kernel_size = SPATIAL_UP_KERNEL_SIZE,
            stride = SPATIAL_UP_STRIDE,
            padding = SPATIAL_UP_PADDING,
        )

        self.spatial_out = nn.Conv3d(
            in_channels = hidden_channels,
            out_channels = OUTPUT_CHANNELS,
            kernel_size = spatial_kernel_size,
            stride = SPATIAL_LOWRES_STRIDE,
            padding = spatial_padding,
        )
    #=================================


    #==================================================
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X shape:
            [N, T, Xdim, Ydim, Zdim, 7]

        Returns:
            [N, 1, Xdim, Ydim, Zdim, 7]
        """
        if X.ndim != 6:
            raise ValueError(
                f"Expected input of shape [N, T, X, Y, Z, 7], got {tuple(X.shape)}"
            )
        if X.shape[-1] != INPUT_CHANNELS:
            raise ValueError(
                f"Expected last dimension to have size {INPUT_CHANNELS}, got {X.shape[-1]}"
            )

        N, T, Xdim, Ydim, Zdim, C = X.shape

        # ==================================================
        # 1. SPATIAL ENCODER ON EACH TIME SLICE
        # [N, T, X, Y, Z, 7]
        # -> [N*T, 7, X, Y, Z]
        # -> [N*T, hidden_channels, X/2, Y/2, Z/2]
        # ==================================================
        X_spatial = X.reshape(N * T, Xdim, Ydim, Zdim, C)
        X_spatial = X_spatial.permute(0, 4, 1, 2, 3)

        X_spatial = self.spatial_down(X_spatial)
        X_spatial = self.act(X_spatial)

        X_spatial = self.spatial_lowres(X_spatial)
        X_spatial = self.act(X_spatial)

        # low-resolution spatial dimensions
        _, H, Xlow, Ylow, Zlow = X_spatial.shape

        # ==================================================
        # 2. BACK TO TIME-INDEXED FORM
        # [N*T, H, Xlow, Ylow, Zlow]
        # -> [N, T, Xlow, Ylow, Zlow, H]
        # ==================================================
        X_spatial = X_spatial.permute(0, 2, 3, 4, 1)
        X_spatial = X_spatial.reshape(N, T, Xlow, Ylow, Zlow, H)

        # ==================================================
        # 3. TEMPORAL CONV AT LOWER RESOLUTION
        # [N, T, Xlow, Ylow, Zlow, H]
        # -> [N, Xlow, Ylow, Zlow, H, T]
        # -> [N*Xlow*Ylow*Zlow, H, T]
        # ==================================================
        X_temporal = X_spatial.permute(0, 2, 3, 4, 5, 1)
        X_temporal = X_temporal.reshape(N * Xlow * Ylow * Zlow, H, T)

        X_temporal = self.temporal_conv(X_temporal)
        X_temporal = self.act(X_temporal)

        # take final time index
        # [N*Xlow*Ylow*Zlow, H, T] -> [N*Xlow*Ylow*Zlow, H]
        X_temporal = X_temporal[:, :, -1]

        # ==================================================
        # 4. BACK TO LOW-RES 3D GRID
        # [N*Xlow*Ylow*Zlow, H]
        # -> [N, H, Xlow, Ylow, Zlow]
        # ==================================================
        X_dec = X_temporal.reshape(N, Xlow, Ylow, Zlow, H)
        X_dec = X_dec.permute(0, 4, 1, 2, 3)

        # ==================================================
        # 5. UPSAMPLE BACK TO FULL RESOLUTION
        # [N, H, Xlow, Ylow, Zlow]
        # -> [N, H, Xdim, Ydim, Zdim]
        # -> [N, 7, Xdim, Ydim, Zdim]
        # ==================================================
        X_dec = self.spatial_up(X_dec)
        X_dec = self.act(X_dec)

        X_dec = self.spatial_out(X_dec)

        # ==================================================
        # 6. RETURN TO WELL FORMAT
        # [N, 7, Xdim, Ydim, Zdim]
        # -> [N, Xdim, Ydim, Zdim, 7]
        # -> [N, 1, Xdim, Ydim, Zdim, 7]
        # ==================================================
        Y_pred = X_dec.permute(0, 2, 3, 4, 1)
        Y_pred = Y_pred.unsqueeze(1)

        return Y_pred
    #================
#====================
#===============================================================================================================================










#============= RUNNING IT ======================================================================================================
#================
def symmetry_MHD(
        model_name,
        spatial_kernel_size,
        temporal_kernel_size,
        hidden_channels,
        activation,
        learning_rate,
        loss_func,
        epochs,
        delta_threshold,
        train_batch_size,
        valid_batch_size,
        test_batch_size,
        train_shuffle,
        eval_shuffle,
        plotting: bool,
        valid_orig,
        test_orig,
        **kwargs,
    ):
    """
    Input data is a time-evolving 64 x 64 x 64 grid of plasma
    that obeys MHD equations.

    A spatial 3-D CNN is run, then a temporal 1-D CNN over the
    augmented data, modified to exploit translational equivariance,
    rotational equivariance, and B-field parity.
    """
    model = SpatioTemporalCNN(
        spatial_kernel_size = spatial_kernel_size,
        temporal_kernel_size = temporal_kernel_size,
        hidden_channels = hidden_channels,
        activation = activation,
    )

    _, train_aug = train_generator()

    return TVT_MHD(
        model = model,
        model_name = model_name,
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = learning_rate,
        ),
        loss_func = loss_func,
        epochs = epochs,
        delta_threshold = delta_threshold,
        train_loader = DataLoader(
            train_aug,
            batch_size = train_batch_size,
            shuffle = train_shuffle,
        ),
        valid_loader = DataLoader(
            valid_orig,
            batch_size = valid_batch_size,
            shuffle = eval_shuffle,
        ),
        test_loader = DataLoader(
            test_orig,
            batch_size = test_batch_size,
            shuffle = eval_shuffle,
        ),
        plotting = plotting,
        test_metadata = test_orig.dataset.metadata,
    )
#===========================


#================
def baseline_MHD(
        model_name,
        spatial_kernel_size,
        temporal_kernel_size,
        hidden_channels,
        activation,
        learning_rate,
        loss_func,
        epochs,
        delta_threshold,
        train_batch_size,
        valid_batch_size,
        test_batch_size,
        train_shuffle,
        eval_shuffle,
        plotting: bool,
        valid_orig,
        test_orig,
        **kwargs,
    ):
    """
    Baseline model:
    same spatiotemporal CNN architecture, but trained on the
    original non-augmented data.
    """
    model = SpatioTemporalCNN(
        spatial_kernel_size = spatial_kernel_size,
        temporal_kernel_size = temporal_kernel_size,
        hidden_channels = hidden_channels,
        activation = activation,
    )

    train_orig, _ = train_generator()

    return TVT_MHD(
        model = model,
        model_name = model_name,
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = learning_rate,
        ),
        loss_func = loss_func,
        epochs = epochs,
        delta_threshold = delta_threshold,
        train_loader = DataLoader(
            train_orig,
            batch_size = train_batch_size,
            shuffle = train_shuffle,
        ),
        valid_loader = DataLoader(
            valid_orig,
            batch_size = valid_batch_size,
            shuffle = eval_shuffle,
        ),
        test_loader = DataLoader(
            test_orig,
            batch_size = test_batch_size,
            shuffle = eval_shuffle,
        ),
        plotting = plotting,
        test_metadata = test_orig.dataset.metadata,
    )
#===========================


#=========================
if __name__ == "__main__":
    valid_orig, test_orig = heldout_generator()

    symmetry_results = symmetry_MHD(
        valid_orig = valid_orig,
        test_orig = test_orig,
        **symmetry_MHD_config,
    )
    baseline_results = baseline_MHD(
        valid_orig = valid_orig,
        test_orig = test_orig,
        **baseline_MHD_config,
    )

    print("\nSymmetry architecture:")
    print("Test loss:", symmetry_results["test_loss"])
    print("VRMSE per field:", symmetry_results["vrmse_per_field"])
    print("Mean VRMSE:", symmetry_results["vrmse_mean"])

    print("\nBaseline architecture:")
    print("Test loss:", baseline_results["test_loss"])
    print("VRMSE per field:", baseline_results["vrmse_per_field"])
    print("Mean VRMSE:", baseline_results["vrmse_mean"])
#=======================================================
#===============================================================================================================================