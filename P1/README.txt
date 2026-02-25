Professor = David
TA = Clark
print(f'Hello {Professor} or {TA}')

1. 
    1. Done.

    2. Done.

    3. P1.py gives (top to bottom):
DOWNLOADS & SETUP: downloading the data, rng-seeding labels, masking out non-RGB stars.
TVT: train, validate, and test processes for n-layer NNs, 
    outputting test loss, train and validate loss plots, parity plot of test data.
KNN(): KNN.
LINEAR REGRESSION: LR.
MULTI-LAYER PERCEPTRON: MLP.
LR, MLP HYPERPARAMETER SWEEPS: nested for-loops over hyperparameters to optimize them.
TUNED SINGLE HYPERPARAMETER SET: used results from hyperparameter sweeps to generate final plots.
USER CALLS: Deciding on single runs or hyperparameter sweeps. @Professor @TA, this is where you
            can mess with the hyperparameters.



2. 
    1. Done.

    2. As discussed in class, minimization of the training loss doesn't correlate with 
       minimization of the validation loss, therefore, all three methods (KNN, LR, MLP)
       had k, weights and biases respectively minimized with respect to the training loss,
       but the stopping condition was based upon the validation loss. To tune the
       hyperparameters for LR and MLP, the functions were called within for-loops to check 
       over various combinations over ranges of their hyperparameters, with the minimum loss
       hyperparameters being chosen.

    3. Methodology described in 2, with the addition that I ran the hyperparameter sweeps,
       and as each minimum-loss hyperparameter set came out, I ran the next sweep with a more 
       narrow range of values around the previously minimizing hyperparameters. Results of test 
       set can be seen in the parity plots for each method, saved within the P1 folder. For the 
       plots currently uploaded, they correspond to the "Single Runs" section of USER CALLS.
       test loss for KNN, LR, and MLP were: 0.0546, 0.310, 0.099, respectively. I was surprised
       by the consistent order-of-magnitude improvement of KNN over LR. Overall, learning-rate 
       seemed to have the largest influence of all the hyperparameters, with epoch varying 
       between hypersweeps, and dim_hidden consistently being best when the first hidden layer
       dimension was around 128. Also, interestingly, the double descent curve can be seen in 
       the LinearRegression_loss.png plot.


Citation: ChatGPT 5.2 was used for general discussions in understanding conceptually the 
pseudocode for the various models, as well as troubleshooting. It generated the KNN plot, 
and helped in "Data prep:" sections by ensuring data was in the correct form. The rest was 
my code.