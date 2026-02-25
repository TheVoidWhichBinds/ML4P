Professor = David
TA = Clark
print(f'Hello {Professor} or {TA}')

1. 
    1. Done.

    2. Done.

    3. P1.py gives (top to bottom):
DOWNLOADS & SETUP: downloading the data, rng-seeding labels, masking out non-RGB stars
TVT: train, validate, and test processes for n-layer NNs, 
    outputting test loss, train and validate loss plots, parity plot of test data.
KNN(): KNN
linear_regression(): LR
MLP(): MLP
FUNCTION CALLS: where I tweaked the hyperparameters, mainly for LR and MLP




2. 
    1. Done.

    2. As discussed in class, minimization of the training loss doesn't correlate with 
       minimization of the validation loss, therefore, all three methods (KNN, LR, MLP)
       had k, weights and biases respectively minimized with respect to the training loss,
       but the stopping condition was based upon the validation loss. To tune the
       hyperparameters for LR and MLP, the functions were called within for-loops to check 
       over various combinations over ranges of their hyperparameters, with the minimum loss
       hyperparameters being chosen.

    3. Methodology described in 2. Results of test set can be seen in the parity plots for
       each method, saved within the P1 folder.


Citation: ChatGPT 5.2 was used for general discussions in understanding conceptually the 
pseudocode for the various models, as well as troubleshooting. It generated the KNN plot, 
and helped in "Data prep:" sections by ensuring data was in the correct form. The rest was 
my code.