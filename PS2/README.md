
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
DATA INITIALIZATION:
MODEL CLASS AND FORWARD FUNCTION:
HYPERPARAMETERS AND RUNNING IT:
AUGMENTATION (augment.py):
TVT_MHD (TVT.py):

                                Part 1
(a) Dataset = MHD_64 - DESCRIPTION
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



LLM Usage:
    