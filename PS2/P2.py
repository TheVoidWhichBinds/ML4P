from pathlib import Path
from the_well.data import WellDataset
from the_well.utils.download import well_download
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from TVT import TVT_MHD
from torch.utils.data import Dataset, ConcatDataset
from augment import Augmentation, b_parity, rotation_symmetry










#------------------- HYPERPARAMETERS ---------------------------------------------------------------------------------------
epochs = 100
delta_threshold = 1e-4
learning_rate = 1e-3
batch_fractions = [0.25, 0.25, 0.25] # Fraction of total train, validate, and testing (respectively) data per batch
dim_hidden = [128, 64, 28] # nodes in each hidden layer # DELETE???
activation = nn.LeakyReLU() 
loss_func = nn.MSELoss()
#---------------------------------------------------------------------------------------------------------------------------










#--------- DATA INITIALIZATION ----------------------------------------------------------------------------------------------
#------------------
# Downloading data:
base_path = Path("./datasets")
dataset_name = "MHD_64"

print("cwd:", Path.cwd())
print("base_path:", base_path.resolve())

for split in ["train", "valid", "test"]:
    split_dir = base_path / dataset_name / "data" / split
    if not split_dir.exists():
        well_download(
            base_path=str(base_path),
            dataset=dataset_name,
            split=split,
        )
#-------------------------


#----------------------------------------
# Splitting data into train, valid, test:
datasets = {} # initializing dictionary
for split in ['train', 'valid', 'test']:
    datasets[split] = WellDataset( # generating key-value pairs
        well_base_path = str(base_path),
        well_dataset_name = dataset_name,
        well_split_name = split,
        n_steps_input = 4, 
        n_steps_output = 1,
        use_normalization = False,
    )
#---------------------------------


#----------------------------------
# Data augmentation:
train_orig = datasets['train']
train_mag = Augmentation(train_orig, transform=b_parity)
train_z90 = Augmentation(
    train_orig,
    transform = lambda sample: rotation_symmetry(sample, axis_rot="z", num_90_rot=1)
)
train_x90 = Augmentation(
    train_orig,
    transform=lambda sample: rotation_symmetry(sample, axis_rot="x", num_90_rot=1)
)

# Combining augmentations:
train_aug = ConcatDataset([train_orig, train_mag, train_z90, train_x90])

# Validation and test data:
valid_orig = datasets['valid']
test_orig = datasets['test']
#----------------------------------------------------------------------------------------------------------------------------










#----------------------------------------------------------------------------------------------------------------------------
class SpatioTemporalCNN(nn.Module):
    """
    Model for Well MHD_64 data with input shape:
        X : [N, T, X, Y, Z, 7]

    Pipeline:
      1. Spatial CNN: convolve over (X,Y,Z) independently for each time slice
      2. Temporal CNN: convolve over T independently at each voxel
      3. Return output in Well-style format:
           [N, 1, X, Y, Z, 7]

    Assumes n_steps_output = 1.
    """
    def __init__(
        self,
        spatial_kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        activation: nn.Module | None = None,
    ):
        super().__init__()

        if activation is None:
            activation = nn.ReLU()
        self.act = activation

        #----------------------------------
        # To preserve dimensions for odd kernel sizes:
        spatial_padding = spatial_kernel_size // 2
        temporal_padding = temporal_kernel_size // 2
        #----------------------------------

        #==================================
        # SPATIAL CNN
        # 4 convolutions: 7 -> 16 -> 32 -> 16 -> 7
        # Input to this block:  [N*T, 7, X, Y, Z]
        # Output from this block: [N*T, 7, X, Y, Z]
        #==================================
        self.spatial_conv1 = nn.Conv3d(
            in_channels=7,
            out_channels=16,
            kernel_size=spatial_kernel_size,
            stride=1,
            padding=spatial_padding,
        )
        self.spatial_conv2 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=spatial_kernel_size,
            stride=1,
            padding=spatial_padding,
        )
        self.spatial_conv3 = nn.Conv3d(
            in_channels=32,
            out_channels=16,
            kernel_size=spatial_kernel_size,
            stride=1,
            padding=spatial_padding,
        )
        self.spatial_conv4 = nn.Conv3d(
            in_channels=16,
            out_channels=7,
            kernel_size=spatial_kernel_size,
            stride=1,
            padding=spatial_padding,
        )

        #==================================
        # TEMPORAL CNN
        # Convolves over time only.
        # Input to this block:  [N*X*Y*Z, 7, T]
        # Output from this block: [N*X*Y*Z, 7, T]
        #==================================
        self.temporal_conv1 = nn.Conv1d(
            in_channels=7,
            out_channels=16,
            kernel_size=temporal_kernel_size,
            stride=1,
            padding=temporal_padding,
        )
        self.temporal_conv2 = nn.Conv1d(
            in_channels=16,
            out_channels=7,
            kernel_size=temporal_kernel_size,
            stride=1,
            padding=temporal_padding,
        )
        #----------------------------------


    #------------------------------------------------------------------------------------------------
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X shape expected:
            [N, T, Xdim, Ydim, Zdim, 7]

        Returns:
            Y_pred with shape [N, 1, Xdim, Ydim, Zdim, 7]
        """
        #----------------------------------
        # Checking input shape:
        if X.ndim != 6:
            raise ValueError(
                f"Expected input of shape [N, T, X, Y, Z, 7], but got shape {tuple(X.shape)}"
            )
        if X.shape[-1] != 7:
            raise ValueError(
                f"Expected 7 channels in last dimension, but got {X.shape[-1]}"
            )
        #----------------------------------

        #----------------------------------
        # Original shape:
        # X = [N, T, Xdim, Ydim, Zdim, 7]
        N, T, Xdim, Ydim, Zdim, C = X.shape
        #----------------------------------

        #================================================================================================
        # 1. RESTRUCTURE FOR SPATIAL CNN
        #
        # Conv3d expects:
        #   [batch, channels, D, H, W]
        #
        # We want to treat each time slice as its own batch element:
        #   [N, T, X, Y, Z, 7] -> [N*T, 7, X, Y, Z]
        #================================================================================================
        X_spatial = X.reshape(N * T, Xdim, Ydim, Zdim, C)      # [N*T, X, Y, Z, 7]
        X_spatial = X_spatial.permute(0, 4, 1, 2, 3)           # [N*T, 7, X, Y, Z]

        #-------------
        # Spatial CNN:
        X_spatial = self.spatial_conv1(X_spatial)
        X_spatial = self.act(X_spatial)

        X_spatial = self.spatial_conv2(X_spatial)
        X_spatial = self.act(X_spatial)

        X_spatial = self.spatial_conv3(X_spatial)
        X_spatial = self.act(X_spatial)

        X_spatial = self.spatial_conv4(X_spatial)
        # X_spatial: [N*T, 7, X, Y, Z]
        #----------------------------------

        #================================================================================================
        # 2. TAKE SPATIAL OUTPUT BACK TO TIME-INDEXED FORM
        #
        # [N*T, 7, X, Y, Z] -> [N, T, X, Y, Z, 7]
        #================================================================================================
        X_spatial = X_spatial.permute(0, 2, 3, 4, 1)           # [N*T, X, Y, Z, 7]
        X_spatial = X_spatial.reshape(N, T, Xdim, Ydim, Zdim, 7)

        #================================================================================================
        # 3. RESTRUCTURE FOR TEMPORAL CNN
        #
        # We now want Conv1d over time only.
        # At each voxel (x,y,z), we take its T-step history and convolve over T.
        #
        # Start:
        #   [N, T, X, Y, Z, 7]
        #
        # Rearrange to:
        #   [N, X, Y, Z, 7, T]
        #
        # Then collapse (N,X,Y,Z) into batch:
        #   [N*X*Y*Z, 7, T]
        #================================================================================================
        X_temporal = X_spatial.permute(0, 2, 3, 4, 5, 1)       # [N, X, Y, Z, 7, T]
        X_temporal = X_temporal.reshape(N * Xdim * Ydim * Zdim, 7, T)

        #----------------------------------
        # Temporal CNN:
        X_temporal = self.temporal_conv1(X_temporal)
        X_temporal = self.act(X_temporal)

        X_temporal = self.temporal_conv2(X_temporal)
        # X_temporal: [N*X*Y*Z, 7, T]
        #----------------------------------

        #================================================================================================
        # 4. TAKE TEMPORAL OUTPUT BACK TO WELL-STYLE FORM
        #
        # Since n_steps_output = 1, we take the final time index after temporal convolution.
        #
        # [N*X*Y*Z, 7, T] -> last time slice -> [N*X*Y*Z, 7]
        # -> [N, X, Y, Z, 7]
        # -> [N, 1, X, Y, Z, 7]
        #================================================================================================
        X_temporal = X_temporal[:, :, -1]                       # [N*X*Y*Z, 7]
        Y_pred = X_temporal.reshape(N, Xdim, Ydim, Zdim, 7)    # [N, X, Y, Z, 7]
        Y_pred = Y_pred.unsqueeze(1)                            # [N, 1, X, Y, Z, 7]

        return Y_pred
    #------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------




def MHD(
        plotting: bool
):
    """
    """

    model = SpatioTemporalCNN(
        activation=activation,
    )

    TVT_MHD(
        model=model,
        model_name='Aug',
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
        loss_func=loss_func,
        epochs=epochs,
        delta_threshold=delta_threshold,
        train_loader=DataLoader(train_aug, batch_size=8, shuffle=True),
        valid_loader=DataLoader(valid_orig, batch_size=8, shuffle=False),
        test_loader=DataLoader(test_orig, batch_size=8, shuffle=False),
        plotting=plotting,
    )


#-------------------------------------------------------------------------------------------------------------------------------------










# # #------------------- RUNNING IT ------------------------------------------------------------------------------------------------------

# # #-------------------------------------------------------------------------------------------------------------------------------------