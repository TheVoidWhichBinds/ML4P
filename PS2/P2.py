from pathlib import Path
from the_well.data import WellDataset
from the_well.utils.download import well_download
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from TVT import TVT_CNN
from torch.utils.data import ConcatDataset









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
base_path = Path("./")
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
for split in ["train", "valid", "test"]:
    datasets[split] = WellDataset( # generating key-value pairs
        well_base_path = str(base_path),
        well_dataset_name = dataset_name,
        well_split_name = split,
        n_steps_input = 4,
        n_steps_output = 1,
        use_normalization = False,
    )
#---------------------------------
#----------------------------------------------------------------------------------------------------------------------------










#--------- DATA AUGMENTATION ------------------------------------------------------------------------------------------------
#---------------
# Original data:
train_orig = datasets["train"]
valid_orig = datasets["valid"]
test_orig = datasets["test"]
#---------------------------

print(train_orig.metadata.field_names)
#--------------
# Rotated data:
train_rot = datasets["train"]
valid_rot = datasets["valid"]
test_rot = datasets["test"]
#---------------------------

#--------------
# B -> -B data:
train_B = datasets["train"]
valid_B = datasets["valid"]
test_B = datasets["test"]
#----------------------------


#--------------------
# Combining datasets:
train_aug = ConcatDataset([train_orig, train_rot, train_B])
valid_aug = ConcatDataset([valid_orig, valid_rot, valid_B])
test_aug  = ConcatDataset([test_orig, test_rot, test_B])
#-------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------










#------------------- CNN ----------------------------------------------------------------------------------------------------
def MHD( plotting: bool):
    """
    CNN that emulates plasma MHD equations with the assistance of 
    augmented data enforcing translation, rotation, and B-field parity.
    """

    #--------------------
    class CNN(nn.Module):
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

        def forward(self, x: torch.Tensor) -> torch.Tensor: 
            for layer in self.layers[:-1]:
                x = self.act(layer(x))
            x = self.layers[-1](x)
            return x
    #---------------


    #---------------------------
    # dims = 
    model = CNN(activation=activation)

    
   
    #------------------------------------------------------------------


    return TVT_CNN(
        model = CNN,
        model_name = model.__name__,
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate),
        loss_func = loss_func,
        epochs = epochs,
        delta_threshold = delta_threshold,
        train_loader = DataLoader(train_aug, batch_size=8, shuffle=True),
        valid_loader = DataLoader(valid_aug, batch_size=8, shuffle=False),
        test_loader = DataLoader(test_aug, batch_size=8, shuffle=False),
        plotting = plotting
    )
#-------------------------------------------------------------------------------------------------------------------------------------










# #------------------- RUNNING IT ------------------------------------------------------------------------------------------------------

# #-------------------------------------------------------------------------------------------------------------------------------------