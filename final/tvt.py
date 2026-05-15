# tvt.py
#================================================================================================================================================
# Train, Validate, Test functions for inner linear regression and outer MLP
#================================================================================================================================================


from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn








def train():
    """
    1) 3 different Inner_k models are forward-propagated, each with its own k value,
        all 3 models using the same
    """