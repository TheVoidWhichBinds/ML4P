import numpy as np
from astropy.io import fits # You might need to pip install this
import pylab # only needed for verification
from matplotlib import pyplot as plt
import torch
import torchvision





#---------------------------------- DOWNLOADS & SETUP -----------------------------------
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

print(type(train_features))


#------------------------------------------ LINEAR REGRESSION -----------------------------------------
def linear_regression():
    
    x = torch.tensor(train_features, dtype=torch.float32) # features (rank 1)
    y = torch.tensor(train_labels['LOGG'], dtype=torch.float32) # (num_spectra, 1) single-output regression 
    w = torch.tensor(0.5 * np.ones(len(x)), dtype=torch.float32, requires_grad=True) # weights (rank 1)
    b = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) # biase (rank 0)


    def forward_pass(x): # linear regression function model
        f = w * x + b
        return f
    
    def loss(y, f): # mean-squared-error loss function
        l = ((y - f)**2).mean()
        return l


    learning_rate = 0.01 # gradient descent step-size factor
    n_epochs = 100 # number of training runs through depth of network

    for epoch in range(n_epochs): 
        f = forward_pass(x) # predict = forward propagation
        l = loss(y, f) # loss 
        l.backward() # gradients calculated using backward propagation
        with torch.no_grad(): # .nograd() to stop weight-tracking at end of epoch
            w -= learning_rate * w.grad # alpha * dL/dw
        w.grad.zero_() # refreshing weight tracking
        
    


# Linear regression takes the form f = w * X + b:







