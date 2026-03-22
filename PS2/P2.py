from pathlib import Path
from the_well.data import WellDataset
from the_well.utils.download import well_download
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from TVT import TVT_CNN
from torch.utils.data import ConcatDataset
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
valid = datasets['valid']
test = datasets['test']
#----------------------------------------------------------------------------------------------------------------------------










#------------------- Spatial CNN ---------------------------------------------------------------------------------------------
def MHD(
        kernel_size,
        activation,
        plotting: bool,
    
    
    
    ):
    """
    CNN that emulates plasma MHD equations with the assistance of 
    augmented data enforcing translation, rotation, and B-field parity.
    """

    #--------------------
    class spatial_CNN(nn.Module):
        def __init__(
            self,
            dims: list[int],
            activation: nn.Module | None = None
        ):
            super().__init__()
            if activation is None:
                activation = nn.ReLU()
            self.act = activation
            self.layers = nn.ModuleList(
                [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
            )

        def forward(self, X: torch.Tensor) -> torch.Tensor: 
            X = nn.Conv3d(
                in_channels = 7 * ,
                out_channels = 7,
                kernel_size = kernel_size,
                stride = 1
            )
            X = self.act(X)
            X = nn.Conv3d(
                in_channels = 7 *,
                out_channels = 7,
                kernel_size = kernel_size,
                stride = 1
            )
            return X
    #---------------------




    
   
    #------------------------------------------------------------------


    return TVT_CNN(
        model = MHD(activation=activation),
        model_name = MHD.__name__(),
        optimizer = torch.optim.Adam(MHD.parameters(), lr=learning_rate),
        loss_func = loss_func,
        epochs = epochs,
        delta_threshold = delta_threshold,
        train_loader = DataLoader(train_aug, batch_size=8, shuffle=True),
        valid_loader = DataLoader(valid_aug, batch_size=8, shuffle=False),
        test_loader = DataLoader(test_aug, batch_size=8, shuffle=False),
        plotting = plotting
    )
#-------------------------------------------------------------------------------------------------------------------------------------










# # #------------------- RUNNING IT ------------------------------------------------------------------------------------------------------

# # #-------------------------------------------------------------------------------------------------------------------------------------