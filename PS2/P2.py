from pathlib import Path
from the_well.data import WellDataset
from the_well.utils.download import well_download
from torch.utils.data import DataLoader
import torch.nn as nn
import torch





#--------- DATA INITIALIZATION ----------------------------------------------------------------------------------------------
#------------------
# Downloading data:
base_path = str(Path("./datasets"))
dataset_name = "MHD_64"
for split in ["train", "valid", "test"]:
    split_dir = base_path / dataset_name / "data" / split
    if not split_dir.exists():
        well_download(
            base_path = base_path, 
            dataset = dataset_name, 
            split = split)
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


#--------------
# Rotated data:



#---------------------------


#--------------
# B -> -B data:



#----------------------------
#----------------------------------------------------------------------------------------------------------------------------










#------------------- CNN ----------------------------------------------------------------------------------------------------
def MHD(func: Callable[[torch.Tensor], torch.Tensor], plotting: bool):
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
    dims = 
    model = 
    model_name = func.__name__
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    #------------------------------------------------------------------


    return TVT_CNN(
        model = model,
        model_name = model_name,
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate),
        loss_func = nn.MSELoss(),
        epochs = epochs,
        delta_threshold = delta_threshold,
        train_loader = train_loader,
        valid_loader = valid_loader,
        test_loader = test_loader,
        plotting = plotting
    )
#-------------------------------------------------------------------------------------------------------------------------------------










#------------------- RUNNING IT ------------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------------------------------------------