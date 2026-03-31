
The Train, Validate, Test function that I created in PS1 has been moved 
to its own file in PS2 and modified, to be called in the problem files.

#------------------------------ P1.py --------------------------------#
                              Sections
TOY FUNCTIONS: The label generator and some even functions to be used
    for the baseline vs Z2 test.
DATA PREP: Decides the amount of data to use, the subdivision into 
    training, validation, and testing, and the toy function choice.
HYPERPARAMETERS: Taken outside the function so that the baseline vs
    Z2 test is fair, i.e. only the squaring of the data before being
    passed into the NN is different between the two.
VARIABLE MLP: Multi-Layer Perceptron that takes func as an argument,
    which serves to modify the incoming feature tensor. For the 
    baseline case, x -> x. For the Z2-symmetry-preserving case,
    x -> x**2. This preserves the f(x) = f(-x) symmetry of even
    functions, but forces symmetry for non-even functions, causing
    the Z2 version to perform well specifically on functions that 
    have this symmetry built-in. Improvement of the test loss relative 
    to the baseline ranged from about the same, to one order of
    magnitude improvement, testing using different seeds, and all 4
    optional even functions (cos, cosh, even poly, exp(x**2)).
RUNNING IT: baseline: standard MLP. Z2: Z2-enforcing function. Also
    contains the actual function calls. 
TVT_MLP (TVT.py):

                                Part 1



#------------------------------ P2.py --------------------------------#
                               Sections
DATA INITIALIZATION: Data download, number of time-steps per sample, 
    data augmentation (See Part 1, (b)) with subset of data (shit was
    too damn big. Run it on a supercomputer if you want).
MODEL CLASS AND FORWARD FUNCTION: Model was a spatial 3-D CNN then a 
    temporal 1-D CNN, with activations after each layer. Tensors had 
    to be modified to be compatible with Conv3d and Conv1d. 
HYPERPARAMETERS AND RUNNING IT: Calling the model, hyperparameters,
    and running train, validate, test function.
AUGMENTATION (augment.py): augmentation functions: B-field parity,
    and rotation function that can rotate 90, 180 or 270 degrees about 
    the axis of choice.
TVT_MHD (TVT.py): Train, validate, test function, same as TVT_MLP but 
    slightly modified to take in train_aug, valid_orig, test_orig 
    (Dataset objects) and extract their "input_fields" and "output_fields"
    tensors.

                                Part 1
(a) Dataset = MHD_64 - A 64x64x64 spatial grid of plasma with periodic
    boundary conditions that obeys 3 equations of ideal MHD: 
    mass continuity, momentum conservation, and the induction equation.
    The density rho, 3-D vector fields of velocity and magnetic field
    are known at each spatial grid, at HOW MANY time stamps.
(b) The symmetries I am enforcing are: 
    Rotational symmetry - the MHD system of equations are equivariant 
        under the rotational symmetry (rho, P, v, B) -> (rho, P, Rv, RB).
        The label tensor is equivariant under this transformation.
    Magnetic field parity - the MHD system of equations are invariant 
        under the Z2 transformation (rho, P, v, B) -> (rho, P, v, -B).
        The label tensor is equivariant under this transformation.
    Translational symmetry - the MHD system of equations are equivariant 
        under the translation (rho(x,t), P(x,t), v(x,t), B(x,t)) -> 
        (rho(x+eps,t), P(x+eps,t), v(x+eps,t), B(x+eps,t)). The label
        tensor is equivariant under this transformation.

                                Part 2
(a) The rotational symmetry and magnetic field parity are enforced via
        augmented data, while the translational symmetry are enforced 
        by the neural network being a CNN.
(b)







LLM Usage: Packaging feature matrices into forms acceptable by Conv3d, Conv1d, 
    