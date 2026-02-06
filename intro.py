import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Folders for tutorials:
def back_to_basics():
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

def autograd():
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

def linear_regression():
    #--------------------- Linear Regression -----------------------#
    # Linear regression: f = w * x + b
    X = torch.tensor([1,2,3,4,5,6,7,8], dtype = torch.float32) # feature tensor
    Y = torch.tensor([2,4,6,8,10,12,14,16], dtype = torch.float32) # labels
    # Gradients wrt weights for optimization needed, w tracking begins:
    # since we have 8 data points, 1 feature per pt, weight tensor = scalar
    w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True) # initializing w

    def forward(x): # model output
        return w * x

    def loss(y, y_pred): # loss = MSE
        return ((y_pred - y)**2).mean()

    x_test = 5.0 # testing data point

    # Training:
    # learning weight = step size (Newton-like) for weights (w = w - alpha * grad(L))
    learning_rate = 0.01 # hyperparameter
    n_epochs = 100 # number of training runs through depth of network

    for epoch in range(n_epochs): 
        y_pred = forward(X) # predict = forward propagation
        l = loss(Y, y_pred) # loss 
        l.backward() # gradients calculated using backward propagation
        with torch.no_grad(): # .nograd() to stop weight-tracking at end of epoch
            w -= learning_rate * w.grad # alpha * dL/dw
        w.grad.zero_() # refreshing weight tracking
        
        if (epoch+1) % 5 == 0:
            print(f'epoch {epoch+1}: w = {w.item():.3f}, loss = {l.item():.3f}')

    print(f'Prediction after training: f({x_test}) = {forward(x_test).item():.3f}')


    #------------------------ PyTorch Linear Regression ------------------------#
    import torch.nn as nn # neural network
    
    # Training samples - expects p > 1 features per data point, therefore shape important!
    X = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8]], dtype=torch.float32)
    Y = torch.tensor([[2],[4],[6],[8],[10],[12],[14],[16]], dtype=torch.float32)
    n_samples, n_features = X.shape # 8 x 1
    print(f'n_samples = {n_samples}, n_features = {n_features}') 

    x_test = torch.tensor([5], dtype=torch.float32) # testing data point

    # Model design:
    # New PyTorch model ALWAYS "inherits (gets same attributes, methods)"" from nn.Module
    class LinearRegression(nn.Module): # linear regression by def has only 1 layer
        def __init__(self, input_dim, output_dim): #
            super(LinearRegression, self).__init__() #
            self.lin = nn.Linear(input_dim, output_dim) #

        def forward(self, x):
            return self.lin(x) # 
        
    input_size, output_size = n_features, n_features

    model = LinearRegression(input_size, output_size)

    loss = nn.MSELoss() # MSE loss function defined
    # Optimizer uses stochastic gradient descent, always inputs model.parameters = weights
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

    #Training loop:
    for epoch in range(n_epochs):
        y_predicted = model(X) # LinearRegression called to run the NN
        l = loss(Y, y_predicted) # loss calcualated
        l.backward() # gradients dL/dw calculated
        optimizer.step() # update weights (model.parameters())
        optimizer.zero_grad() # zero the gradient record after updating

        if (epoch+1) % 5 == 0:
            w, b = model.parameters() # unpack parameters
            print('epoch', epoch+1, ': w =', w[0][0].item(), 'loss =', l.item())

    print(f'Prediction after training: f({x_test.item()}) = {model(x_test).item():.3f}')

def NN():
    # Overview: All training data (or a subset = 'batch') is made into the design
    # matrix X = batch_size x 784 (28x28 screen pixels are flattened into one
    # dimension), then multiplied by 784 x L weight matrix (L is up to me =
    # next layer width / # neurons in layer), and added by bias (batch size x L)


    # Hyperparameters:
    input_size = 784 # features (e.g. # pixels per image)
    hidden_size = 500 # number of neurons in (1) hidden layer
    num_classes = 10 # number of label categories
    num_epochs = 2 # number of runs through the network
    batch_size = 100 # only this many training data run through network at once
    learning_rate = 0.001 # magnitude of step factor of Loss gradient

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform = transforms.ToTensor(), # could be in different form initially
        download=True,
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform = transforms.ToTensor(),
    )

    # DataLoader gives way to iterate over dataset:
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )


    # Fully connected (EVERY input neuron connected to EVERY output neuron)
    # Neural Network construction:
    class NeuralNet(nn.Module):
        # When calling an instance (see model = NeuralNet below), __init__ auto-called
        def __init__(self, input_size, hidden_size, num_classes): 
            super(NeuralNet, self).__init__() # calls nn.Module __init__ func
            # First layer- in: feature #/width, out: hidden layer width
            self.ll = nn.Linear(input_size, hidden_size) 
            self.relu = nn.ReLU() # activation func sigma
            # Prediction- in: hidden layer width, out: number of label categories
            self.l2 = nn.Linear(hidden_size, num_classes)

        # Running forward pass:
        def forward(self, x): 
            out = self.ll(x)
            out = self.relu(out)
            out = self.l2(out)
            # No activation or softmax at the end (for specific loss func input below)
            return out
        
    # Instance:
    model = NeuralNet(input_size, hidden_size, num_classes)

    # Loss and Optimizer:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training model:
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1,28*28) # flattening pixels into 1 dimension
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward() # dL/dw
            optimizer.step() # w -= learning_rate * dL/dw
            optimizer.zero_grad() # refresh gradient tracking

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step {i+1}/{n_total_steps}], Loss: {loss}')

    
    # Test model:
    with torch.no_grad(): # gradient tracking not needed for test
        n_correct = 0 # initializing number of correct test data prediction vs target
        n_samples = len(test_loader.dataset)

        for images, labels in test_loader:
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # max returns (output_value, index)
            n_correct += (predicted == labels).sum().item() # checking predicted output index matches label index
            
        acc = n_correct / n_samples # accuracy ratio
        print(f'Accuracy of the network on the {n_samples} test images: {100*acc}%')




NN()
