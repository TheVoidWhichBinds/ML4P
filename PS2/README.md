
The Train, Validate, Test function that I created in PS1 has been moved 
to its own file in PS2 and modified, to be called in the problem files.

#------------------------------ P1.py --------------------------------#
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


#------------------------------ P2.py --------------------------------#

