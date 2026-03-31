from pathlib import Path
from the_well.data import WellDataset
from the_well.utils.download import well_download
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from TVT import TVT_MHD
from torch.utils.data import ConcatDataset, Subset
from .augment import Augmentation, b_parity, rotation_symmetry










#============= ADJUSTABLE HYPERPARAMETERS =====================================================================================
#================================================================
# Hyperparameters that both architectures must have identical for
#   a fair baseline vs symmetry-enforced comparison:

# Subset settings:
# Multiplication factor of train_aug_divisor should match number 
#   of concatenated data sets in DATA INITIALIZATION s.t. 
#   baseline NN and symmetry-enforcing NN both are given the same 
#   qty of data
train_orig_divisor = 100
train_aug_divisor = 5 * train_orig_divisor 
valid_divisor = 10
test_divisor = 10

# Batch settings:
train_batch_size = 4
valid_batch_size = 2
test_batch_size = 2

# Model architecture settings:
spatial_kernel_size = 3
temporal_kernel_size = 3
hidden_channels = 4
activation = nn.LeakyReLU()

# Optimizer settings:
learning_rate = 1e-3
loss_func = nn.MSELoss()
epochs = 15
delta_threshold = 1e-2
#==================


#====================
#--------------------
# Both architectures:
architecture_config = {
    # DataLoader settings:
    "train_batch_size": train_batch_size,
    "valid_batch_size": valid_batch_size,
    "test_batch_size": test_batch_size,
    "train_shuffle": True,
    "eval_shuffle": False,

    # Model architecture settings:
    "spatial_kernel_size": spatial_kernel_size,
    "temporal_kernel_size": temporal_kernel_size,
    "hidden_channels": hidden_channels,
    "activation": activation,

    # Optimizer / training settings:
    "learning_rate": learning_rate,
    "loss_func": loss_func,
    "epochs": epochs,
    "delta_threshold": delta_threshold,
    "plotting": True,
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

for split in ["train", "valid", "test"]:
    split_dir = base_path / "MHD_64" / "data" / split
    if not split_dir.exists():
        well_download(
            base_path = str(base_path),
            dataset = "MHD_64",
            split = split,
        )
#=========================


#========================================
# Splitting data into train, valid, test:
datasets = {} # initializing dictionary
for split in ["train", "valid", "test"]:
    datasets[split] = WellDataset( # generating key-value pairs
        well_base_path = str(base_path),
        well_dataset_name = "MHD_64",
        well_split_name = split,
        n_steps_input = 4,
        n_steps_output = 1,
        use_normalization = False,
    )
#=================================


#=======================
def dataset_generator():
    """
    Augments data, and takes subsets of the datasets.
    """
    # Data augmentation:
    train_orig = datasets["train"]
    train_mag = Augmentation(train_orig, transform=b_parity)
    train_z180 = Augmentation(
        train_orig,
        transform = lambda sample: rotation_symmetry(sample, axis_rot="z", num_90_rot=2)
    )
    train_x90 = Augmentation(
        train_orig,
        transform=lambda sample: rotation_symmetry(sample, axis_rot="x", num_90_rot=1)
    )
    train_y270 = Augmentation(
        train_orig,
        transform=lambda sample: rotation_symmetry(sample, axis_rot="y", num_90_rot=3)
    )
    #=================================================================================


    #===============================
    # Combining train augmentations:
    train_aug = ConcatDataset([train_orig, train_mag, train_z180, train_x90, train_y270])
    N_train_aug = len(train_aug)
    indices = torch.randperm(N_train_aug)[:(N_train_aug // train_aug_divisor)]
    train_aug = Subset(train_aug, indices)

    # Train data for baseline:
    N_train_orig = len(train_orig)
    indices = torch.randperm(N_train_orig)[:(N_train_orig // train_orig_divisor)]
    train_orig = Subset(train_orig, indices)

    # Validation data:
    valid_orig = datasets["valid"]
    N_valid = len(valid_orig)
    indices = torch.randperm(N_valid)[:(N_valid // valid_divisor)]
    valid_orig = Subset(valid_orig, indices)

    # Test data:
    test_orig = datasets["test"]
    N_test = len(test_orig)
    indices = torch.randperm(N_test)[:(N_test // test_divisor)]
    test_orig = Subset(test_orig, indices)


    print("Data augmentation complete ...")
    return train_orig, train_aug, valid_orig, test_orig
    #==================================================
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
        spatial_kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        hidden_channels: int = 16,
        activation: nn.Module | None = None,
    ):
        super().__init__()

        if activation is None:
            activation = nn.ReLU()
        self.act = activation

        spatial_padding = spatial_kernel_size // 2
        temporal_padding = temporal_kernel_size // 2

        # -----------------------------
        # SPATIAL ENCODER
        # [N*T, 7, 64, 64, 64]
        # -> [N*T, hidden_channels, 32, 32, 32]
        # -----------------------------
        self.spatial_down = nn.Conv3d(
            in_channels = 7,
            out_channels = hidden_channels,
            kernel_size = spatial_kernel_size,
            stride = 2,
            padding = spatial_padding,
        )

        # Optional extra processing at low resolution
        self.spatial_lowres = nn.Conv3d(
            in_channels = hidden_channels,
            out_channels = hidden_channels,
            kernel_size = spatial_kernel_size,
            stride = 1,
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
            stride = 1,
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
            kernel_size = 4,
            stride = 2,
            padding = 1,
        )

        self.spatial_out = nn.Conv3d(
            in_channels = hidden_channels,
            out_channels = 7,
            kernel_size = spatial_kernel_size,
            stride = 1,
            padding = spatial_padding,
        )
    #=====================


    #========================
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
        if X.shape[-1] != 7:
            raise ValueError(
                f"Expected last dimension to have size 7, got {X.shape[-1]}"
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

    _, train_aug, valid_orig, test_orig = dataset_generator()


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

    train_orig, _, valid_orig, test_orig = dataset_generator()


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
    )
#===========================


#=========================
if __name__ == "__main__":
    symmetry_results = symmetry_MHD(**symmetry_MHD_config)
    print(symmetry_results)

    baseline_results = baseline_MHD(**baseline_MHD_config)
    print(baseline_results)
#==========================
#===============================================================================================================================