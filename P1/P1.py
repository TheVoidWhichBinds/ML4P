# NOTES:
# Consider Layer Normalization, Weight Normalization, Normalization Propagation



import numpy as np
from astropy.io import fits 
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor
# TensorFlow/Keras imports removed because this script uses PyTorch for models.
# If you need Keras functionality, install TensorFlow and uncomment the lines below.
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense



#------------------- DOWNLOADS & SETUP ------------------------------------------------
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
#----------------------------------------------------------------------------------------------------------------
  







#------------------- K NEAREST NEIGHBORS ------------------------------------------
def KNN(k_range: list[int]):
    """
    K-Nearest Neighbors Algorithm. Operates differently than 
    Linear Regression or MLP. Finds the k that minimizes the
    loss function.
    ------- Params ----------
    k_range: list[int]
        list of k values to average over
    """
    #---------------------------------------------------------
    # Data prep:
    x_train = train_features.astype(np.float32)
    y_train = np.array(train_labels["LOGG"], dtype=np.float32)
    #
    x_valid = valid_features.astype(np.float32)
    y_valid = np.array(valid_labels["LOGG"], dtype=np.float32)
    #
    x_test = test_features.astype(np.float32)
    y_test = np.array(test_labels["LOGG"], dtype=np.float32)
    #---------------------------------------------------------


    #------------------------------------------------------------------------------------------------
    # Train:
    def train():
        #
        loss_train = np.empty(len(k_range), dtype=np.float32) # initializing training loss for each k

        for i, k in enumerate(k_range):
            # Running KNN:
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(x_train, y_train) 
            f_train = knn.predict(x_train) 
            loss_train[i] = ((y_train - f_train)**2).mean() # MSE loss
            #
            i_min = int(np.argmin(loss_train))  # index of minimum loss over k_range
            k_opt = k_range[i_min] # best training k

        return k_opt
    

    # Validate:
    def validate():
        # Running KNN:
        k_opt = train(k_range) # extracting the optimal k from the training runs
        knn = KNeighborsRegressor(n_neighbors=k_opt)
        knn.fit(x_valid, y_valid)
        f_valid = knn.predict(x_valid) 
        loss_valid = ((y_valid - f_valid)**2).mean() 

        return k_opt, loss_valid


    # Test:
    # def test(k_range: list[int]):
    #      # Running KNN:
    #     k_opt = train(k_range) # extracting the optimal k from the training runs
    #     knn = KNeighborsRegressor(n_neighbors=k_opt)
    #     knn.fit(x_test, y_test)
    #     f_test = knn.predict(x_test) 
    #     loss_test = ((y_train - f_test)**2).mean() 
        
    #     return print(f'Test MSE loss:{loss_test}')

    return validate(k_range)
    #-----------------------------------------------------------------------------------------------
    
# Running K-Nearest-Neighbors with validation loss return:
# knn_results = KNN(np.arange(2,10,1)) # storing k_opt (from training) and validation loss
# print(f"optimal k = {knn_results[0]}, validation loss = {knn_results[1]:.5g}")
  







#------------------- LR & MLP DATA PREP --------------------------------------------------------------------------
# Data prep:
x_train = torch.tensor(train_features, dtype=torch.float32) # (N, d)
y_np = np.array(train_labels["LOGG"], dtype=np.float32)  # forces native float32
y_train = torch.from_numpy(y_np).view(-1, 1)
train_batch_size = int(N_train/4)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=train_batch_size, shuffle=True)  # batches
    
x_valid = torch.tensor(valid_features, dtype=torch.float32) # (N, d)
y_np = np.array(valid_labels["LOGG"], dtype=np.float32)  # forces native float32
y_valid = torch.from_numpy(y_np).view(-1, 1)
valid_batch_size = int(N_valid/4)
valid_loader = DataLoader(TensorDataset(x_valid, y_valid), batch_size=valid_batch_size, shuffle=False)  # batches
    
x_test = torch.tensor(test_features, dtype=torch.float32) # (N, d)
y_np = np.array(test_labels["LOGG"], dtype=np.float32)  # forces native float32
y_test = torch.from_numpy(y_np).view(-1, 1)
test_batch_size = int(N_test/4)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=test_batch_size, shuffle=False)  # batches
#-----------------------------------------------------------------------------------------------------------------
  






def train_and_validate(            
        model,                    
        optimizer,                
        loss_func,                
        epochs,                   
        delta_threshold           
        ):
    """
    For Linear Regression and Multi-Layer Perceptron. Not KNN
    Contains train function, validate function, epoch loop and 
    train and validation loss plotting.
    """
 
    def train():
        """
        Uses batches to minimize training loss.
        ------- Params ------------------------
        train_loader:
            Training data subdivided into batches of prescribed size
        optimizer:
            Optimizer used to minimize training loss as a function of weights and biases
        model:
            Custom model (LR, MLP) for hidden layers and activations
        loss_func:
            Chosen metric to decide loss, e.g. mean squared error 
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
        
        loss_avg = loss_total/N

        return model, loss_avg
    


    def validate(model, valid_loader):
        """
        Runs validation data thru network using training parameters.
        ------ Params ----------------------------------------------
        valid_loader:
            Validation data subdivided into batches of prescribed size
        model:
            Model with updated params from training run
        loss_func:
            Same loss function used in train()
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
            
        loss_avg = loss_total/N

        return loss_avg
    


    # Looping over epochs, stopping either at training or validation threshold loss:
    loss_array = np.empty((epochs, 2)) # initializing array that tracks train and valid loss
    N_epochs = int(0) # initializing epochs looped over

    for epoch in range(epochs): 
        model, loss_train = train(train_loader) # 1 training update step
        loss_valid = validate(model, valid_loader) # validation loss using updated train step params
        #
        loss_array[epoch, 0] = loss_train 
        loss_array[epoch, 1] = loss_valid
        # Stop condition based off validation loss to prevent overfitting:
        avg_size = 10  # number of epochs to average over 
        #
        if epoch + 1 >= 2 * avg_size: # checking for stop condition only after 2 * avg_size
            curr_avg = loss_array[epoch-avg_size+1: epoch+1, 1].mean()
            prev_avg = loss_array[epoch-2*avg_size+1: epoch-avg_size+1, 1].mean()
            #
            if abs(curr_avg - prev_avg) <= delta_threshold:
                break
    
        N_epochs += 1



    def plot_loss(out_dir="./P1"):
        """
        Plots train and validation loss as a function of epoch.
        """
        model_name = model.__class__.__name__
        plt.figure()
        plt.title(f'{model_name} Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Curves')
        plt.scatter(np.arange(1, N_epochs + 1), loss_array[:N_epochs, 0], s=4, label='Training Loss')
        plt.scatter(np.arange(1, N_epochs + 1), loss_array[:N_epochs, 1], s=4, label='Validation Loss')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'{out_dir}/{model_name}_loss.png')


    
    def test(test_loader):
        """

        """
        with torch.no_grad():
            for xb, yb in test_loader:
                f = model(xb) # forward pass
                loss_batch = loss_func(f, yb) # loss averaged over batch (1/B)
                #
                loss_total += loss_batch.item() * xb.size(0)  # sum over batch sum of squares
                N += xb.size(0) # sum over number of samples in each batch
            
        loss_avg = loss_total/N

        return loss_avg
















#------------------- LINEAR REGRESSION ------------------------------------------
def linear_regression(epochs, delta_threshold, learning_rate):
    
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
    #--------------------------------------------------

    #-------------------------------------------------------------------------------
    # Training and validating:
    model = LinearRegression(input_dim=x_train.shape[1], output_dim=1)

    train_and_validate(
        model = model,
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate),
        loss_func = nn.MSELoss(),
        epochs = epochs,
        delta_threshold = delta_threshold
    )

linear_regression(100, 1e-3, 1e-4)
    

    

    




#------------------- MULTI-LAYER PERCEPTRON -------------------------------------------------------------------
def MLP(epochs, delta_threshold, learning_rate):

    #----------------------------------------------------------------------------------------------------
    # Generating MLP class:
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
    #----------------------------------------------------------------------------------------------------



    #-------------------------------------------------------------------------------
    # Regression Prep:
    dim_input = x_train.shape[1] # feature dimension = dimension of input layer
    dim_output = 1 # label dimension = dimension of output layer
    model = MultiLayerPerceptron([dim_input, 128, 64, dim_output], activation=nn.ReLU())
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #--------------------------------------------------------------------------------



    #-------------------------------------------------------------------------------------------------------
    


    
MLP(400, 128, 1e-5, 1e-4)
