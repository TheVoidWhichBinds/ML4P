import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
project_dir = Path(__file__).resolve().parent
base_path = project_dir / 'P2'









#=================== TEST, VALIDATE, TRAIN FUNCTION for MLPs =============================================================
def TVT_MLP(            
        model,
        model_name,                    
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
    For MLPs
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

    #===========
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
    #===================================================================


    #==============
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
    #======================


    #===============================================================================
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
    #================


    #==========
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
    #======================================
    
    loss_test, y_true, y_pred = test()

    #===================
    if plotting == True:
        # Plotting training and validation loss:
        plt.figure()
        plt.title(f'{model_name} Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Curves')
        plt.scatter(np.arange(1, N_epochs + 1), loss_array[:N_epochs, 0], s=4, label='Training Loss')
        plt.scatter(np.arange(1, N_epochs + 1), loss_array[:N_epochs, 1], s=4, label='Validation Loss')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'./PS2/{model_name}_loss.png')
      
        # Parity Plot (predicted test data vs true test labels):
        plt.figure()
        plt.title(f'{model_name} Parity Plot')
        plt.xlabel("Labels y")
        plt.ylabel("Test Prediction ŷ")
        plt.scatter(y_true, y_pred, s=2, alpha=0.15)
        mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        plt.plot([mn, mx], [mn, mx], color='black')
        plt.savefig(f'./PS2/{model_name}_parity.png')
        #============================================

    return float(loss_test)
#=========================================================================================================================










#============== TEST, VALIDATE, TRAIN FUNCTION for CNNs ===================================================================
def TVT_MHD(            
        model,
        model_name,                    
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
    For CNNs applied to problem 2
    Contains train function, validate function, epoch loop for both,
    and test function
    ------- Params ------------------------------------------------
        model:
            Custom model for hidden layers and activations
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
            Plots train and validation loss
        ------- Returns --------------------------------------
        loss_test: float
            test loss averaged over batches
    """

    #===========
    def train():
        """
        Uses PyTorch to forward and back propagate weights and biases
        to minimize training data loss.
        """
        model.train()
        loss_total = 0.0
        N = 0.0
        i = 0

        for batch in train_loader: # loop over batches
            print(f'Training batch #{i} ...')
            xb = batch['input_fields']
            yb = batch['output_fields']
            optimizer.zero_grad() # refreshing gradient-tracker
            f = model(xb) # forward pass
            loss_batch = loss_func(f, yb) # loss averaged over batch (1/B)
            loss_batch.backward() # backward pass
            optimizer.step() # updating weights and biases
            #
            loss_total += loss_batch.item() * xb.size(0)  # sum over batch sum of squares
            N += xb.size(0) # sum over number of samples in each batch
            i += 1

        return model, loss_total / N # model with updated params, avg loss
    #===================================================================


    #==============
    def validate():
        """
        Runs validation data thru network using training parameters.
        """
        model.eval()
        loss_total = 0.0
        N = 0.0
        i = 0

        # Running with training weights and biases on validation data, without updating:
        with torch.no_grad():
            for batch in valid_loader:
                print(f'Validation batch #{i}')
                xb = batch['input_fields']
                yb = batch['output_fields']
                f = model(xb) # forward pass
                loss_batch = loss_func(f, yb) # loss averaged over batch (1/B)
                #
                loss_total += loss_batch.item() * xb.size(0)  # sum over batch sum of squares
                N += xb.size(0) # sum over number of samples in each batch
                i += 1

        return loss_total / N
    #======================


    #===============================================================================
    # Looping over epochs, stopping either at training or validation threshold loss:
    loss_array = np.empty((epochs, 2)) # initializing array that tracks train and valid loss
    N_epochs = int(0) # initializing epochs looped over

    for epoch in range(epochs):
        print(f'Train and validate run, epoch #{epoch+1}')
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
    #================


    #==========
    def test():
        """
        Takes final weights and biases from train and validation epoch loops
        and calculates test loss.
        """
        print('Beginning test ...')
        model.eval()
        loss_total = 0.0
        N = 0.0

        with torch.no_grad():
            for batch in test_loader:
                xb = batch['input_fields']
                yb = batch['output_fields']
                f = model(xb)
                loss_batch = loss_func(f, yb)
                #
                loss_total += loss_batch.item() * xb.size(0)
                N += xb.size(0)

        return loss_total / N
    #======================================


    print('Beginning test run ...')
    loss_test = test()

    #===================
    if plotting == True:
        # Plotting training and validation loss:
        plt.figure()
        plt.title(f'{model_name} Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Curves')
        plt.scatter(np.arange(1, N_epochs + 1), loss_array[:N_epochs, 0], s=4, label='Training Loss')
        plt.scatter(np.arange(1, N_epochs + 1), loss_array[:N_epochs, 1], s=4, label='Validation Loss')
        plt.yscale('log')
        plt.legend()
        plt.savefig(base_path / f'{model_name}_loss.png')
        #============================================

    return float(loss_test)
#==========================================================================================================================