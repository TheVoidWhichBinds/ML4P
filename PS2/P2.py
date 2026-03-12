import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from collections.abc import Callable
from TVT import TVT_CNN









#------------------- CNN -------------------------------------------------------------------------------------------------------------
def MHD(func: Callable[[torch.Tensor], torch.Tensor], plotting: bool):
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