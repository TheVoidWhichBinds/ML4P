# inner_model.py
#================================================================================================================================================
# "inner" model that generates operators with sequentially more timestamps.
#================================================================================================================================================

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


class ProgressiveLocalMap():
    """
    Constructs T - 1 operators, with each operator acting on the
    2nd-to-last timestamp to map to the last timestamp, but with
    progressively more history of previous timestamps kept.
    """

