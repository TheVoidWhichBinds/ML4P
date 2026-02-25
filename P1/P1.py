
import numpy as np
from astropy.io import fits 
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor










#------------------- DOWNLOADS & SETUP ---------------------------------------------------------------------------
# Downloading labels:
path_labels = "./data/labels.fits"
allstar = fits.open(path_labels) # loads star database data
labels = allstar[1].data # labels

# 'ANDs' RGB (True) with check of label params compared to threshold values (True/False)
# True + True = True, True + False = False
# Selecting Red Giant Branch:
RGB = True
RGB = np.logical_and(RGB, labels['TEFF'] > 3500.) 
RGB = np.logical_and(RGB, labels['TEFF'] < 5400.)
RGB = np.logical_and(RGB, labels['LOGG'] < 3.0)
RGB = np.logical_and(RGB, labels['LOGG'] > 0.0)
RGB = np.logical_and(RGB, labels['H'] < 10.5)
RGB_labels = labels[RGB] # selects labels that meet criteria

# Making train, validation, and test data sets:
rng = np.random.default_rng(17) # arbitrary seed to shuffle indices of RGB dataset (DO NOT CHANGE)
N_RGB = len(RGB_labels) # number of labels that meet criteria
N_train, N_valid, N_test = 1024, 256, 512 # number of each type of data
I = rng.permutation(N_RGB) # shuffling post-criteria dataset

# Assigning the shuffled indices to each type of data:
I_train = I[0:N_train] 
I_valid = I[N_train:N_train+N_valid] 
I_test = I[N_train+N_valid:N_train+N_valid+N_test]

# Train, validate, test: 
train_labels = RGB_labels[I_train]
valid_labels = RGB_labels[I_valid]
test_labels = RGB_labels[I_test]

# Downloading features: 
train_features = np.load('./data/train_features.npy')
valid_features = np.load('./data/valid_features.npy')
test_features = np.load('./data/test_features.npy')
# Normalizing/standardizing features:
mu = train_features.mean(axis=0, keepdims=True)
sig = train_features.std(axis=0, keepdims=True) + 1e-8
train_features = (train_features - mu) / sig
valid_features = (valid_features - mu) / sig
test_features  = (test_features  - mu) / sig
#-----------------------------------------------------------------------------------------------------------------










#------------------- TRAIN, VALIDATE, TEST FUNCTION -----------------------------------------------------------
def TVT(            
        model,                    
        optimizer,                
        loss_func: nn.Module,                
        epochs: int,                   
        delta_threshold: float,
        train_loader,
        valid_loader,
        test_loader,
        plotting: bool           
        ):
    """
    For Linear Regression and Multi-Layer Perceptron. Not KNN.
    Contains train function, validate function, epoch loop over
    both, train and validation loss plotting, test function, 
    test data parity plot.
    ------- Params ------------------------
        model:
            Custom model (LR, MLP) for hidden layers and activations
        optimizer:
            Optimizer used to minimize training loss as a function 
            of weights and biases
        loss_func:
            Chosen metric to decide loss, e.g. mean squared error
        epochs: int
            Number of times the NN gets run
        delta_threshold: float
            Stopping threshold for change in validation loss, averaged 
            over epochs
        train_loader, valid_loader, test_loader:
            Data subdivided into batches of prescribed size
        plotting: bool
            Plots train and validation loss, test parity plots
        ------- Returns --------------------------------------
        loss_test: float
            test loss averaged over batches
    """
    #-----------
    def train():
        """
        Uses PyTorch to forward and back propagate weights and biases
        to minimize training data loss.
        """
        model.train()
        loss_total = 0.0
        N = 0.0
        
        for xb, yb in train_loader: # loop over batches
            optimizer.zero_grad() # refreshing gradient-tracker
            f = model(xb) # forward pass
            loss_batch = loss_func(f, yb) # loss averaged over batch (1/B)
            loss_batch.backward() # backward pass
            optimizer.step() # updating weights and biases
            #
            loss_total += loss_batch.item() * xb.size(0)  # sum over batch sum of squares
            N += xb.size(0) # sum over number of samples in each batch
        
        return model, loss_total/N # model with updated params, avg loss
    #-------------------------------------------------------------------


    #--------------
    def validate():
        """
        Runs validation data thru network using training parameters.
        """
        model.eval()
        loss_total = 0.0
        N = 0.0

        # Running with training weights and biases on validation data, without updating:
        with torch.no_grad():
            for xb, yb in valid_loader:
                f = model(xb) # forward pass
                loss_batch = loss_func(f, yb) # loss averaged over batch (1/B)
                #
                loss_total += loss_batch.item() * xb.size(0)  # sum over batch sum of squares
                N += xb.size(0) # sum over number of samples in each batch

        return loss_total/N
    #----------------------------------------------------------------------------------------


    #---------------------------------------------------------------------------------------
    # Looping over epochs, stopping either at training or validation threshold loss:
    loss_array = np.empty((epochs, 2)) # initializing array that tracks train and valid loss
    N_epochs = int(0) # initializing epochs looped over

    for epoch in range(epochs): 
        model, loss_train = train() # 1 training update step
        loss_valid = validate() # validation loss using updated train step params
        loss_array[epoch, 0] = loss_train 
        loss_array[epoch, 1] = loss_valid
        #
        N_epochs += 1 
        #
        avg_size = 10  # number of epochs to average over 
        if epoch + 1 >= 2 * avg_size: # checking for stop condition only after 2 * avg_size
            curr_avg = loss_array[epoch-avg_size+1: epoch+1, 1].mean()
            prev_avg = loss_array[epoch-2*avg_size+1: epoch-avg_size+1, 1].mean()
            #
            if abs(curr_avg - prev_avg) <= delta_threshold:
                break
    #----------------


    #----------
    def test():
        """
        Takes final weights and biases from test and validation epoch loops
        and calculates test loss.
        """
        model.eval()
        loss_total, N = 0.0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for xb, yb in test_loader:
                f = model(xb)
                loss_total += loss_func(f, yb).item() * xb.size(0)
                N += xb.size(0)
                y_true.append(yb)
                y_pred.append(f)

        y_true = torch.cat(y_true).squeeze().numpy()
        y_pred = torch.cat(y_pred).squeeze().numpy()
        return loss_total/N, y_true, y_pred
    #--------------------------------------
    
    loss_test, y_true, y_pred = test()

    #---------------------------------------
    if plotting == True:
        # Plotting training and validation loss:
        model_name = model.__class__.__name__
        plt.figure()
        plt.title(f'{model_name} Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Curves')
        plt.scatter(np.arange(1, N_epochs + 1), loss_array[:N_epochs, 0], s=4, label='Training Loss')
        plt.scatter(np.arange(1, N_epochs + 1), loss_array[:N_epochs, 1], s=4, label='Validation Loss')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'./P1/{model_name}_loss.png')
      
        # Parity Plot (predicted test data vs true test labels):
        plt.figure()
        plt.title(f'{model_name} Parity Plot')
        plt.xlabel("Labels y")
        plt.ylabel("Test Prediction ŷ")
        plt.scatter(y_true, y_pred, s=6)
        mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        plt.plot([mn, mx], [mn, mx])
        plt.savefig(f'./P1/{model_name}_parity.png')
        #-------------------------------------------
    

    return float(loss_test)
#----------------------------------------------------------------------------------------------------------------










#------------------- K NEAREST NEIGHBORS --------------------------------------------------------------------------
def KNN(k_range: list[int]):
    #-----------
    # Data prep:
    x_train = train_features.astype(np.float32)
    y_train = np.array(train_labels["LOGG"], dtype=np.float32)
    #
    x_valid = valid_features.astype(np.float32)
    y_valid = np.array(valid_labels["LOGG"], dtype=np.float32)
    #
    x_test = test_features.astype(np.float32)
    y_test = np.array(test_labels["LOGG"], dtype=np.float32)
    #-------------------------------------------------------


    #-----------------
    # Regression Prep:
    model = KNeighborsRegressor()
    #----------------------------


    #------------------------------
    # Train + Validation loss vs k:
    loss_train = np.empty(len(k_range), dtype=np.float32)
    loss_valid = np.empty(len(k_range), dtype=np.float32)

    for i, k in enumerate(k_range):
        model.set_params(n_neighbors=int(k))
        model.fit(x_train, y_train)
        #
        f_train = model.predict(x_train)
        loss_train[i] = ((y_train - f_train) ** 2).mean()
        #
        f_valid = model.predict(x_valid)
        loss_valid[i] = ((y_valid - f_valid) ** 2).mean()

    i_best = int(np.argmin(loss_valid))
    k_opt = int(k_range[i_best])

    plt.figure()
    plt.title("KNN Training & Validation Loss vs k")
    plt.xlabel("k")
    plt.ylabel("MSE")
    plt.plot(k_range, loss_train, marker="o", linewidth=1, label="Training MSE")
    plt.plot(k_range, loss_valid, marker="o", linewidth=1, label="Validation MSE")
    plt.axvline(k_opt, linestyle="--")
    plt.yscale("log")
    plt.legend()
    plt.savefig("./P1/KNN_loss.png")
    #-------------------------------
   

    #-------------
    # Parity plot:
    model.set_params(n_neighbors=k_opt)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    loss_test = ((y_test - y_pred) ** 2).mean()

    plt.figure()
    plt.title(f"KNN Parity Plot (k={k_opt})")
    plt.xlabel("Labels y")
    plt.ylabel("Test Prediction ŷ")
    plt.scatter(y_test, y_pred, s=6)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx])
    plt.savefig("./P1/KNN_parity.png")
    print(f'KNN final average test loss = {loss_test:.5g},' \
          f' with optimal k value = {k_opt}')
    #--------------------------------------
#----------------------------------------------------------------------------------------------------------------










#------------------- LINEAR REGRESSION ---------------------------------------------------------------------------
def linear_regression(
        epochs: int, 
        delta_threshold: float, 
        learning_rate: float, 
        batch_fractions: list[float],
        plotting: bool
    ):
    """
    Linear Regression ML Model.
    ------ Params -------------
    epochs: int
        number of epochs to run for each batch
    delta_threshold: float
        change in delta_loss that must be met to stop
    learning_rate: float
        scaling for dL/dw in optimization (w += learning_rate * dL/dw)
    batch_fraction: list[float]
        fraction of total data in each batch for [train, validate, test] data
    plotting: bool
        Plots train and validation loss, test parity plots
    """

    #-----------------------------
    # Error conditions & defaults:
    if len(batch_fractions) != 3:
        raise ValueError('batch_fractions must contain train, validate, and test fractions')
    for frac in batch_fractions:
        if frac <= 0 or frac > 1:
            raise ValueError('Values in batch_fractions must be between 0 and 1')
    #----------------------------------------------------------------------------


    #-----------
    # Data prep:
    x_train = torch.tensor(train_features, dtype=torch.float32) # (N, d)
    y_np = np.array(train_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_train = torch.from_numpy(y_np).view(-1, 1)
    train_batch_size = max(1, int(N_train * batch_fractions[0]))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=train_batch_size, shuffle=True) 
        
    x_valid = torch.tensor(valid_features, dtype=torch.float32) # (N, d)
    y_np = np.array(valid_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_valid = torch.from_numpy(y_np).view(-1, 1)
    valid_batch_size = max(1, int(N_valid * batch_fractions[1]))
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=valid_batch_size, shuffle=False)  
        
    x_test = torch.tensor(test_features, dtype=torch.float32) # (N, d)
    y_np = np.array(test_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_test = torch.from_numpy(y_np).view(-1, 1)
    test_batch_size  = max(1, int(N_test  * batch_fractions[2]))
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=test_batch_size, shuffle=False)  
    #-------------------------------------------------------------------------------------------------
    

    #-------------------------------------------------
    # Generating linear regression class from PyTorch:
    class LinearRegression(nn.Module):
        #
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.lin = nn.Linear(input_dim, output_dim)
        #
        def forward(self, x):
            return self.lin(x)
    #-------------------------


    #-----------------
    # Regression prep:
    model = LinearRegression(input_dim=x_train.shape[1], output_dim=1)
    #-----------------------------------------------------------------


    #-----------------------
    # Train, Validate, Test:
    return TVT(
        model = model,
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate),
        loss_func = nn.MSELoss(),
        epochs = epochs,
        delta_threshold = delta_threshold,
        train_loader = train_loader,
        valid_loader = valid_loader,
        test_loader = test_loader,
        plotting = plotting
    )
    #----------------------------
#----------------------------------------------------------------------------------------------------------------

    

    






#------------------- MULTI-LAYER PERCEPTRON ---------------------------------------------------------------------
def MLP(
    epochs: int, 
    delta_threshold: float, 
    learning_rate: float, 
    batch_fractions: list[float],
    dim_hidden: list[int],
    plotting: bool,
    seed: int | None = None,
    activation: nn.Module | None = None
    ):
    """
    Multi-layer perceptron ML model.
    ------ Params ------------------
    epochs: int
        number of epochs to run for each batch
    delta_threshold: float
        change in delta_loss that must be met to stop
    learning_rate: float
        scaling for dL/dw in optimization (w += learning_rate * dL/dw)
    batch_fractions: list[float]
        fraction of total data in each batch for [train, validate, test] data
    dim_hidden: list[int]
        number of nodes in each hidden layer
    plotting: bool
        Plots train and validation loss, test parity plots
    seed: int
        seed to shuffle the training data, initial weights and biases for problem 2.
    activation: PyTorch module
        activation function to be used
    """

    #-----------------------------
    # Error conditions & defaults:
    if len(batch_fractions) != 3:
        raise ValueError('batch_fractions must contain train, validate, and test fractions')
    for frac in batch_fractions:
        if frac <= 0 or frac > 1:
            raise ValueError('Values in batch_fractions must be between 0 and 1')
    #----------------------------------------------------------------------------


    #-----------
    # Data prep:
    x_train = torch.tensor(train_features, dtype=torch.float32) # (N, d)
    y_np = np.array(train_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_train = torch.from_numpy(y_np).view(-1, 1)
    train_batch_size = max(1, int(N_train * batch_fractions[0]))
    g = None
    if seed is not None:
        torch.manual_seed(seed) # randomizer for initial weights and biases
        g = torch.Generator() # shuffling training batches
        g.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), 
        batch_size=train_batch_size, 
        shuffle=True,
        generator=g
    ) 
        
    x_valid = torch.tensor(valid_features, dtype=torch.float32) # (N, d)
    y_np = np.array(valid_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_valid = torch.from_numpy(y_np).view(-1, 1)
    valid_batch_size = max(1, int(N_valid * batch_fractions[1]))
    valid_loader = DataLoader(
        TensorDataset(x_valid, y_valid), 
        batch_size=valid_batch_size, 
        shuffle=False
    )  
        
    x_test = torch.tensor(test_features, dtype=torch.float32) # (N, d)
    y_np = np.array(test_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_test = torch.from_numpy(y_np).view(-1, 1)
    test_batch_size  = max(1, int(N_test  * batch_fractions[2]))
    test_loader = DataLoader(
        TensorDataset(x_test, y_test), 
        batch_size=test_batch_size, 
        shuffle=False
    )  
    #----------------
    

    #-------------------------------------
    class MultiLayerPerceptron(nn.Module):
        #
        def __init__(self, dims:list[int], activation: nn.Module | None = None):
            super().__init__()
            """
            Creates the layers with dimensions given by dims
            ----- Parameters -------------------------------
            dims: list of integers
                Dimension of each layer, including input and output
            activation: PyTorch Module
                activation function to be applied to each layer
            """
            if activation is None:
                activation = nn.ReLU() # default activation function
            # .layers is a list of modules (linear regression) between layers:
            self.layers = nn.ModuleList( 
                [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
            )
            self.act = activation # activation function


        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass - runs data thru layers
            ------ Parameters ------------------
            x: tensor
                Input data
            ------ Returns ------
            x: tensor
                Output of NN - estimate of label y
            """
            for layer in self.layers[:-1]: # for each layer (up to 2nd to last) ...
                x = self.act(layer(x)) # push the input data through a layer, input into activation func
            x = self.layers[-1](x) # 2nd to last to output (label) doesn't get activation func
            return x
    #---------------


    #-----------------
    # Regression Prep:
    dim_input = x_train.shape[1] # feature dimension = dimension of input layer
    dim_output = 1 # label dimension = dimension of output layer
    dims = [dim_input] + dim_hidden + [dim_output]
    model = MultiLayerPerceptron(dims, activation=activation)
    #-----------------------------------------------------------------------------------


    #-----------------------
    # Train, Validate, Test:
    return TVT(
        model = model,
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate),
        loss_func = nn.MSELoss(),
        epochs = epochs,
        delta_threshold = delta_threshold,
        train_loader = train_loader,
        valid_loader = valid_loader,
        test_loader = test_loader,
        plotting = plotting
    )
    #----------------------------
#----------------------------------------------------------------------------------------------------------------










#------------------- LR, MLP HYPERPARAMETER SWEEPS  --------------------------------------------------------------------
#-------------------
def LR_hypersweep(
        epochs_options: list[int],
        learning_rate_options: list[float],
        batch_fractions_options: list[list[float]],
    ):
    """
    Runs LR for combination of hyperparameters: epochs, 
    learning rate, batch fractions of total data for training, 
    validation and test data
    """
    loss_LR = [] # initializing test loss dict list
    # Looping over all hyperparameter options:
    for epochs in epochs_options:
        for learning_rate in learning_rate_options:
            for batch_fractions in batch_fractions_options:
                # Calling LR func:
                loss_test = linear_regression( 
                    epochs=epochs,
                    delta_threshold=1e-4,
                    learning_rate=learning_rate,
                    batch_fractions=batch_fractions,
                    plotting=False
                )
                # Creating dictionary keys:
                run = { 
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_fractions": batch_fractions,
                    "test_loss": float(loss_test),
                }
                loss_LR.append(run) # appending list of dictionaries

    best_LR = min(loss_LR, key=lambda d: d["test_loss"])

    print("\nBest Linear Regression run (by test loss):")
    print(
        f"test_loss = {best_LR['test_loss']:.5g}, "
        f"epochs = {best_LR['epochs']}, "
        f"learning_rate = {best_LR['learning_rate']:.1e}, "
        f"batch_fractions = {best_LR['batch_fractions']}"
    )
#--------------------------------------------------------


#--------------------
def MLP_hypersweep(
        epochs_options: list[int],
        learning_rate_options: list[float],
        batch_fractions_options: list[list[float]],
        dim_hidden_options: list[list[int]],
        activation: nn.Module
    ):
    """
    Runs MLP for combination of hyperparameters: epochs, 
    learning rate, batch fractions of total data for training, 
    validation and test data, # of neurons in each hidden layer
    """
    loss_MLP = [] # initializing test loss dict list
    # Looping over all hyperparameter options:
    for epochs in epochs_options:
        for learning_rate in learning_rate_options:
            for batch_fractions in batch_fractions_options:
                for dim_hidden in dim_hidden_options:
                    # Calling MLP func:
                    loss_test = MLP(
                        epochs=epochs,
                        delta_threshold=1e-4,
                        learning_rate=learning_rate,
                        batch_fractions=batch_fractions,
                        dim_hidden=dim_hidden,
                        plotting=False,
                        seed=None,
                        activation=activation
                    )
                    # Creating dictionary keys:
                    run = {
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "batch_fractions": batch_fractions,
                        "dim_hidden": dim_hidden,
                        "activation": "activation",
                        "seed": None,
                        "test_loss": float(loss_test),
                    }
                    loss_MLP.append(run) # appending list of dictionaries

    best_MLP = min(loss_MLP, key=lambda d: d["test_loss"])

    print("\nBest MLP run (by test loss):")
    print(
        f"test_loss = {best_MLP['test_loss']:.5g}, "
        f"epochs = {best_MLP['epochs']}, "
        f"learning_rate = {best_MLP['learning_rate']:.1e}, "
        f"batch_fractions = {best_MLP['batch_fractions']}, "
        f"dim_hidden = {best_MLP['dim_hidden']}, "
        f"activation = {best_MLP['activation']}, "
        f"seed = {best_MLP['seed']}"
    )
    #-------------------------------
#------------------------------------------------------------------------------------------------------------------------










#------------------- TUNED SINGLE HYPERPARAMETER SET  --------------------------------------------------------------------
# K-Nearest-Neighbors run:
#------------------------------------------------
def KNN_run(): # no hyperparameter tuning for KNN
    return KNN(k_range = np.arange(2,25,1))
#------------------------------------------


# Linear Regression run:
#-----------------------
def LR_single(
        epochs: int,
        delta_threshold: float,
        learning_rate: float,
        batch_fractions: list[float],
        plotting: bool
    ):
    loss_test = linear_regression( 
                    epochs=epochs,
                    delta_threshold=delta_threshold,
                    learning_rate=learning_rate,
                    batch_fractions=batch_fractions,
                    plotting=plotting
                )
    return(print(f'LR averaged test loss = {loss_test:.5g}'))
#------------------------------------------------------------


# Multi-Layer Perceptron run:
#----------------------------
def MLP_single(
        epochs: int,
        delta_threshold: float,
        learning_rate: float,
        batch_fractions: list[float],
        dim_hidden: list[int],
        plotting: bool,
        seed: int | None,
        activation: nn.Module | None = None
    ):
    loss_test = MLP(
                    epochs=epochs,
                    delta_threshold=delta_threshold,
                    learning_rate=learning_rate,
                    batch_fractions=batch_fractions,
                    dim_hidden=dim_hidden,
                    plotting=plotting,
                    seed=seed,
                    activation=activation
                )
    return(print(f'MLP averaged test loss = {loss_test:.5g}'))
#-------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------










#------------------- USER CALLS  -----------------------------------------------------------------------------------------
# Grandmaster control for if single runs (tuned hyperparams, plotting) = True
# or hyperparameter sweep (hyperparam options, NO plotting) = False:
single = True

# Single Runs:
#-----------------
if single == True:
    KNN_run()

    LR_single(
        epochs = 800,
        delta_threshold = 1e-4,
        learning_rate = 2e-3,
        batch_fractions = [0.25,0.25,0.25],
        plotting = False
    )
    
    MLP_single(
        epochs = 500,
        delta_threshold = 1e-4,
        learning_rate = 5.2e-4,
        batch_fractions = [0.1,0.1,0.1],
        dim_hidden = [128,100,64,28],
        plotting = False,
        seed = None,
        activation = nn.LeakyReLU()
    )
#-----------------------------


# Hyperparameter sweep:
#----------------------
else:
    KNN_run()

    LR_hypersweep(
        epochs_options = [800, 900, 1000], 
        learning_rate_options = [1e-3, 2e-3, 3e-3],
        batch_fractions_options = [[0.25, 0.25, 0.25],
                                   [0.2, 0.2, 0.2],
                                   [0.1, 0.1, 0.1]]
    )

    MLP_hypersweep(
        epochs_options = [500, 600],
        learning_rate_options = [4.8e-4, 5e-4, 5.2e-4],
        batch_fractions_options = [[0.25, 0.25, 0.25],
                                   [0.2, 0.2, 0.2],
                                   [0.1, 0.1, 0.1]],
        dim_hidden_options = [[100, 150, 75],
                              [100, 150, 100, 50],
                              [128, 100, 64, 28]],
        activation = nn.ReLU()
    )
#------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------