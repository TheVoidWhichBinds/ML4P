

#------------------------------ P1.py --------------------------------#
                              Sections
HYPERPARAMETERS: Taken outside the function so that the baseline vs
    Z2 test is fair, i.e. only the squaring of the data before being
    passed into the NN is different between the two.
TOY FUNCTIONS: The label generator and some even functions to be used
    for the baseline vs Z2 test.
DATA PREP: Decides the amount of data to use, the subdivision into 
    training, validation, and testing, and the toy function choice.
VARIABLE MLP: Multi-Layer Perceptron that takes func as an argument,
    which serves to modify the incoming feature tensor. For the 
    baseline case, x -> x. For the Z2-symmetry-preserving case,
    x -> x**2. This preserves the f(x) = f(-x) symmetry of even
    functions, but forces symmetry for non-even functions, causing
    the Z2 version to perform well specifically on functions that 
    have this symmetry built-in. 
RUNNING IT: baseline: standard MLP. Z2: Z2-enforcing function. Also
    contains the actual function calls. 
TVT_MLP (TVT.py): Test, Validate, and Training function for an MLP 
    architecture.


#------------------------------ P2.py --------------------------------#
                              Sections
ADJUSTABLE HYPERPARAMETERS: All tweakable hyperparameters. See code 
    comments for each one's meaning.
DATA INITIALIZATION: Data download, func to hold validation and test
    data consistent for both architectures, data augmentation 
    (See Part 1, (b)) with subset of data (shit was too damn big. Run 
    it on a supercomputer if you want).
MODEL: Model was a spatial 3-D CNN, with stride to donwsample grid 
    resolution, then a temporal 1-D CNN, with activations after each 
    layer, then a resampling back up to the 64^3 grid. Tensors had to 
    be modified to be compatible with Conv3d and Conv1d. 
RUNNING IT: Calls architecture functions, prints resulting test loss.
AUGMENTATION (augment.py): augmentation functions: B-field parity,
    and rotation function that can rotate 90, 180 or 270 degrees about 
    the axis of choice.
TVT_MHD (TVT.py): Train, validate, test function, same as TVT_MLP but 
    slightly modified to take in train_aug, valid_orig, test_orig 
    (Dataset objects) and extract their "input_fields" and "output_fields"
    tensors.





    