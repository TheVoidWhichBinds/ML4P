import numpy as np
import torch

#------------- Basics + NumPy parallels ------------#
# Torch is tensor based, therefore applicable to GPUs
# as well as CPUs!

# Same general functionality as NumPy, e.g.
x = torch.rand(5,3)
#        .empty(),
#        .zeros(),
#        .ones(),
#        .add(x,y) (element-wise)
#        .mul(x,y)
#        .div(x,y)
# But with tensors as well
x = torch.empty(4,2,5)

# Tensor has gradient argument (Bool)
# True = PyTorch calculates gradients 
# (e.g. optimization). Default = False
x = torch.tensor([3,1], requires_grad=True)

# Slicing of PyTorch tensors happens as normal

# Reshaping the tensor:
x = torch.randn(4,4)
x_reshaped = x.view(16)

# Conversion between NumPy array and PyTorch tensor:
a = torch.ones(5)
b = a.numpy() # converts to NumPy array CHANGES WITH A
c = torch.tensor(a) # creates a tensor COPY of A

# All tensors default created on CPU, but can be
# moved or created directly to/on GPU!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ------------------------------------------------------------------#




#-------------------- Autograd ---------------------#
# Generally requires requires_grad=True in tensors
x = torch.randn(3, requires_grad=True)
y = x + 2 # all operations to tensor x tracked on computational graph
print(x)
print(y) # y gains attribute grad_fn=<AddBackward0> (back propagation)
z = y * y * 3 # gradient still tacked
print(z)
z = z.mean()
print(z)
# Gradient:
print(x.grad) # prints None because x currently has no gradient
z.backward # calculates gradient of z with respect to original variable, x
print(x.grad) # now prints gradient dz/dx

# EVERY TIME backward() is called, gradient for tensor is accumulated into
# .grad attribute. Neural networks often run with a for-loop. To restart
# the gradient tracker at the end of each iteration then, we use 
# optimizer.zero_grad()

# Prior to updating the weights, or after training, we don't want the
# tensor to track the gradient computation, prevented using:
a.requires_grad_(False) # flag changed in place
a.requires_grad_(True)
b = a.detach() # copies a but with grad flag set to False
with torch.no_grad():
    b = a ** 2 # operation conducted with grad flag set to False


#--------------------- Linear Regression -----------------------#
# Linear regression: f = w * x + b
X = torch.tensor([1,2,3,4,5,6,7,8], dtype = torch.float32)
Y = torch.tensor([2,4,6,8,10,12,14,16], dtype = torch.float32)
 # Gradients wrt weights for optimization needed, w tracking begins:
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x): # model output
    return w * X

def loss(y, y_pred): # loss = MSE
    return ((y_pred - y)**2).mean()

X_test = 5.0 # test sample 5

# Training:
# learning weight = step size (Newton-like) for weights (w = w - alpha * grad(L))
learning_rate = 0.01 
n_epochs = 100 # number of training runs through depth of network

for epoch in range(n_epochs): 
    y_pred = forward(X) # predict = forward propagation
    l = loss(Y, y_pred) # loss 
    l.backward() # gradients calculated using backward propagation
    with torch.no_grad(): # .nograd() to stop weight-tracking at end of epoch
        w -= learning_rate * w.grad # alpha * dL/dw
    w.grad.zero_() # refreshing weight tracking