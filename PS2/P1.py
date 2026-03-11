import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from collections.abc import Callable
from TVT import train_validate_test




#------------------- TOY FUNCTIONS ---------------------------------------------------------------------------------------------------
def toy_labels(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """
    Generates the labels y_dataset based on the function of choice
    """
    return f(x)

def cosine(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(x)

def hyperbolic_cosine(x: torch.Tensor) -> torch.Tensor:
    return torch.cosh(x)

def poly(x: torch.Tensor) -> torch.Tensor:
    return 2*x**6 - 3*x**4 + 10*x**2

def exp_fn(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x**2)
#------------------------------------------------------------------------------------------------------------------------------------




#------------------- DATA PREP ------------------------------------------------------------------------------------------------------
N_dataset = int(1e4) # total number of datapoints to generate
N_train, N_valid, N_test = int(7e3), int(1.5e3), int(1.5e3)

x_dataset = torch.linspace(-4, 4, N_dataset, dtype=torch.float32).view(-1, 1) # all features (converted into (N,1) tensor)

# Shuffling data and assigning indices:
rng = np.random.default_rng(17)
I = rng.permutation(N_dataset)
I_train = I[:N_train]
I_valid = I[N_train:N_train+N_valid]
I_test = I[N_train+N_valid:N_train+N_valid+N_test]

# Dataset range depends on the function chosen, for numerical stability:
x_ranges = {
    cosine: (-4, 4),
    hyperbolic_cosine: (-2, 2),
    poly: (-2, 2),
    exp_fn: (-1.5, 1.5),
}
function = exp_fn # CHOOSE FUNCTION HERE
x_min, x_max = x_ranges[function]
x_dataset = torch.linspace(x_min, x_max, N_dataset, dtype=torch.float32).view(-1, 1) # (N,) array made into (N,1) 2D tensor
y_dataset = toy_labels(function, x_dataset) # generating labels 

# Assigning data to TVT categories:
x_train = x_dataset[I_train]
x_valid = x_dataset[I_valid]
x_test  = x_dataset[I_test]
y_train = y_dataset[I_train]
y_valid = y_dataset[I_valid]
y_test  = y_dataset[I_test]
#-------------------------------------------------------------------------------------------------------------------------------------




#------------------- HYPERPARAMETERS -------------------------------------------------------------------------------------------------
epochs = 100
delta_threshold = 1e-4
learning_rate = 1e-3
batch_fractions = [0.25, 0.25, 0.25] # Fraction of total train, validate, and testing (respectively) data per batch
dim_hidden = [128, 64, 28] # nodes in each hidden layer
seed = 12 # seed for batch shuffling
activation = nn.LeakyReLU() 
#-------------------------------------------------------------------------------------------------------------------------------------










#------------------- GENERIC NN ------------------------------------------------------------------------------------------------------
def h_theta(func: Callable[[torch.Tensor], torch.Tensor], plotting: bool):
    """
    Generic MLP-style NN.
    func = preprocessing map applied once to input.
    Use func(x)=x for baseline, func(x)=x**2 for Z2-even model.
    """
    #----------------------------
    # Catching errors:
    if len(batch_fractions) != 3:
        raise ValueError("batch_fractions must contain train, validate, and test fractions")
    for frac in batch_fractions:
        if frac <= 0 or frac > 1:
            raise ValueError("Values in batch_fractions must be between 0 and 1")

    # Generating seed for training data shuffling:
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    #------------------


    #--------------------
    class MLP(nn.Module):
        def __init__(
            self,
            dims: list[int],
            preprocess: Callable[[torch.Tensor], torch.Tensor],
            activation: nn.Module | None = None
        ):
            super().__init__()
            if activation is None:
                activation = nn.ReLU()
            self.preprocess = preprocess
            self.act = activation
            self.layers = nn.ModuleList(
                [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.preprocess(x)  
            for layer in self.layers[:-1]:
                x = self.act(layer(x))
            x = self.layers[-1](x)
            return x
    #---------------


    #---------------------------
    dim_input = x_train.shape[1]
    dim_output = y_train.shape[1]
    dims = [dim_input] + dim_hidden + [dim_output]
    model = MLP(dims=dims, preprocess=func, activation=activation)
    model_name = func.__name__
    #-------------------------

    #-----------------------------------------------------------
    train_batch_size = max(1, int(N_train * batch_fractions[0]))
    valid_batch_size = max(1, int(N_valid * batch_fractions[1]))
    test_batch_size  = max(1, int(N_test  * batch_fractions[2]))

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=train_batch_size,
        shuffle=True,
        generator=g
    )
    valid_loader = DataLoader(
        TensorDataset(x_valid, y_valid),
        batch_size=valid_batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=test_batch_size,
        shuffle=False
    )
    #----------------


    return train_validate_test(
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
def baseline(x: torch.Tensor) -> torch.Tensor:
    return x

def Z2(x: torch.Tensor) -> torch.Tensor:
    return x**2

baseline_loss = h_theta( # w*x + b in between layers
    func = baseline, 
    plotting = False
) 
Z2_loss = h_theta( # initial x**2 and then w*x + b in between layers 
    func = Z2,
    plotting = False
) 

print(f'Baseline test loss: {baseline_loss:.5e}')
print(f'Z2 test loss: {Z2_loss:.5e}')
#-------------------------------------------------------------------------------------------------------------------------------------