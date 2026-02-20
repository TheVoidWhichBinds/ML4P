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



#------------------------------------------ DOWNLOADS & SETUP ------------------------------------------ 
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








#------------------------------------------ LINEAR REGRESSION ------------------------------------------
def linear_regression(epochs, batch_size, delta_threshold, learning_rate):
    
    #-------------------------------------------------------------------------------
    # Generating linear regression class from PyTorch:
    class LinearRegression(nn.Module):
        #
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.lin = nn.Linear(input_dim, output_dim)
        #
        def forward(self, x):
            return self.lin(x)

    # Data prep:
    x_train = torch.tensor(train_features, dtype=torch.float32) # (N, d)
    y_np = np.array(train_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_train = torch.from_numpy(y_np).view(-1, 1)
    dataset = TensorDataset(x_train, y_train)
    #
    x_valid = torch.tensor(valid_features, dtype=torch.float32) # (N, d)
    y_np = np.array(valid_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_valid = torch.from_numpy(y_np).view(-1, 1)
    #
    x_test = torch.tensor(test_features, dtype=torch.float32) # (N, d)
    y_np = np.array(test_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_test = torch.from_numpy(y_np).view(-1, 1)

    # Regression Prep:
    model = LinearRegression(input_dim=x_train.shape[1], output_dim=1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    #--------------------------------------------------------------------------------



    #------------------------------------------------------------------------------
    # Train:
    def train():

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # dividing train data into batches:
        loss_prev = None

        for epoch in range(epochs): 
            loss_total = 0.0 # initializing sum of squares loss over all data
            N = 0 # initializing sum of number of data 
            for xb, yb in loader: # loop over batches
                optimizer.zero_grad() # refreshing gradient-tracker
                f = model(xb) # forward pass
                loss_batch = loss_fn(f, yb) # loss averaged over batch (1/B)
                loss_batch.backward() # backward pass
                optimizer.step() # updating weights and biases
                #
                loss_total += loss_batch.item() * xb.size(0)  # sum over batch sum of squares
                N += xb.size(0) # sum over number of samples in each batch

            loss_epoch = loss_total / N  # epoch mean loss 

            # Checking delta of epoch loss to terminate loop if threshold met:
            if loss_prev is not None:
                loss_delta = abs(loss_epoch - loss_prev)
                if loss_delta <= delta_threshold:
                    W = model.lin.weight
                    b = model.lin.bias
                    #print(f"STOP epoch {epoch+1}: loss={loss_epoch:.6g}, Δloss={loss_delta:.3g}, ||W||={W.norm().item():.3g}, b={b.item():.3g}")
                    break

            loss_prev = loss_epoch 
        
        return W, b


    # Validate:
    def validate():
        #
        W, b = train()
        f_valid = W * x_valid + b # linear regression with trained weights & biases
        loss_valid = ((y_valid - f_valid)**2).mean() # MSE
        #
        return loss_valid


    # Test:
    # def test():
        
    #     W, b = train()
    #     f_test = W * x_test + b # linear regression with trained weights & biases
    #     loss_test = ((y_test - f_test)**2).mean() # MSE
        
    #     return loss_test


    # Validation loss returned:
    return validate()
    #------------------------------------------------------------------------------


# Running linear regression with validation loss return:
#print(f'linear regression validation loss: {linear_regression(100, 128, 1e-3, 1e-3).item():.5g}') 
    







#---------------------------------------- K NEAREST NEIGHBORS ------------------------------------------
def KNN(k_range):
    
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
    def train(k_range: list[int]):

        loss_train = np.empty(len(k_range), dtype=np.float32) # initializing training loss for each k

        for i, k in enumerate(k_range):
            # Running KNN:
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(x_train, y_train) 
            f_train = knn.predict(x_train) 
            loss_train[i] = ((y_train - f_train)**2).mean() # MSE loss

            i_min = int(np.argmin(loss_train))  # index of minimum loss over k_range
            k_opt = k_range[i_min] # best training k

        return k_opt
    

    # Validate:
    def validate(k_range: list[int]):
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
    

    

    




#--------------------------------------- MULTI-LAYER PERCEPTRON ---------------------------------------
def MLP(epochs, batch_size, delta_threshold, learning_rate):

    #-------------------------------------------------------------------------------
    # Generating MLP class:

    
    # Data prep:
    x_train = torch.tensor(train_features, dtype=torch.float32) # (N, d)
    y_np = np.array(train_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_train = torch.from_numpy(y_np).view(-1, 1)
    dataset = TensorDataset(x_train, y_train)
    #
    x_valid = torch.tensor(valid_features, dtype=torch.float32) # (N, d)
    y_np = np.array(valid_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_valid = torch.from_numpy(y_np).view(-1, 1)
    #
    x_test = torch.tensor(test_features, dtype=torch.float32) # (N, d)
    y_np = np.array(test_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_test = torch.from_numpy(y_np).view(-1, 1)

    # Regression Prep:
    model = LinearRegression(input_dim=x_train.shape[1], output_dim=1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    #--------------------------------------------------------------------------------



    #-------------------------------------------------------------------
    # Train: 
    def train():

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # dividing train data into batches:
        loss_prev = None


        for epoch in range(epochs): 
            loss_total = 0.0 # initializing sum of squares loss over all data
            N = 0 # initializing sum of number of data 
            for xb, yb in loader: # loop over batches
                optimizer.zero_grad() # refreshing gradient-tracker
                f = model(xb) # forward pass
                loss_batch = loss_fn(f, yb) # loss averaged over batch (1/B)
                loss_batch.backward() # backward pass
                optimizer.step() # updating weights and biases
                #
                loss_total += loss_batch.item() * xb.size(0)  # sum over batch sum of squares
                N += xb.size(0) # sum over number of samples in each batch

            loss_epoch = loss_total / N  # epoch mean loss 

            # Checking delta of epoch loss to terminate loop if threshold met:
            if loss_prev is not None:
                loss_delta = abs(loss_epoch - loss_prev)
                if loss_delta <= delta_threshold:
                    W = model.lin.weight
                    b = model.lin.bias
                    #print(f"STOP epoch {epoch+1}: loss={loss_epoch:.6g}, Δloss={loss_delta:.3g}, ||W||={W.norm().item():.3g}, b={b.item():.3g}")
                    break

            loss_prev = loss_epoch 
        
        return W, b