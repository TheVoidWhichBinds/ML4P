import numpy as np
from astropy.io import fits 
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor




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
def linear_regression(epochs, batch_size, delta_threshold):

    #-------------------------------------------------------------------------------
    class LinearRegression(nn.Module):
        #
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.lin = nn.Linear(input_dim, output_dim)
        #
        def forward(self, X):
            return self.lin(X)

    # Data prep:
    X_train = torch.tensor(train_features, dtype=torch.float32) # (N, d)
    y_np = np.array(train_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_train = torch.from_numpy(y_np).view(-1, 1)
    dataset = TensorDataset(X_train, y_train)
    #
    X_valid = torch.tensor(valid_features, dtype=torch.float32) # (N, d)
    y_np = np.array(valid_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_valid = torch.from_numpy(y_np).view(-1, 1)
    #
    X_test = torch.tensor(test_features, dtype=torch.float32) # (N, d)
    y_np = np.array(test_labels["LOGG"], dtype=np.float32)  # forces native float32
    y_test = torch.from_numpy(y_np).view(-1, 1)

    # Regression Prep:
    model = LinearRegression(input_dim=X_train.shape[1], output_dim=1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)
    #--------------------------------------------------------------------------------



    #------------------------------------------------------------------------------
    # Train:
    def train(epochs, batch_size, delta_threshold):

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
    def validate(epochs, batch_size, delta_threshold):
        #
        W, b = train(epochs, batch_size, delta_threshold)
        f_valid = W * X_valid + b # linear regression with trained weights & biases
        loss_valid = ((y_valid - f_valid)**2).mean() # MSE
        #
        return loss_valid


    # Validation loss returned:
    return validate(epochs, batch_size, delta_threshold)
    #------------------------------------------------------------------------------


# Running linear regression with validation loss return:
# print(linear_regression(100, 128, 1e-3)) 
    







#---------------------------------------- K NEAREST NEIGHBORS ------------------------------------------
def KNN(k):
    
    #---------------------------------------------------------
    # Data prep:
    X_train = train_features.astype(np.float32)
    y_train = np.array(train_labels["LOGG"], dtype=np.float32)
    #---------------------------------------------------------



    #-------------------------------------------------------------------------------
    # Train:
    def train(k):

        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train) 
        f_train = knn.predict(X_train) # model predictions for y_train of every star
        
        loss_train = ((y_train - f_train)**2).mean() # MSE
        # print(f'loss: {float(loss_train):.6f}')

    

    # Validate:


    


    

    




#--------------------------------------- MULTI-LAYER PERCEPTRON ---------------------------------------
def MLP():
    xxx = 4
