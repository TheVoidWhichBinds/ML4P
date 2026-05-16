Professor = David
TA = Clark
print(f'Hello {Professor} or {TA}')

README.txt

FILES:

config.py:
    Stores the main settings for the project. This is where the dataset path,
    output folders, k values, model size, Taylor activation settings, training
    settings, pretraining settings, plotting settings, and quick test settings
    are controlled. If you want to mess with the run without editing the main
    code, this is usually the file to change.


data.py:
    Handles the WELL turbulent_radiative_layer_2D data. It finds the train,
    validation, and test files, loads the HDF5 states, normalizes the physical
    channels, pads the boundaries, extracts local 3x3 spatial patches, and builds
    the PyTorch datasets. This is also where the multi-k dataset is made, so the
    same model can train on different amounts of time history.


model.py:
    Defines the neural network model. This includes the Taylor-series activation,
    the InnerK CNN model, and the optional exponential VRMSE-curve slope loss.
    InnerK takes local spatiotemporal patches shaped like [batch, channels, k,
    spatial height, spatial width] and predicts the next center-pixel physical
    state.


run.py:
    Main training file. It picks the device, builds the train and validation
    loaders, computes VRMSE, runs the training and validation epochs, saves the
    checkpoint after each epoch, and writes the train/validation VRMSE log. This
    is the main file for training one shared InnerK model over the selected k
    values.


pretrain.py:
    Optional warm-start step before the full multi-k training run. It trains
    InnerK on only PRETRAIN_K first, then saves a pretraining checkpoint that can
    be loaded by run.py if warm starting is turned on in config.py.


test.py:
    Runs the saved InnerK checkpoint on the held-out test split. It reports the
    final test VRMSE, saves the test metrics to a CSV log, and saves parity-plot
    data comparing predicted values against target values.


plot.py:
    Makes the output plots from the saved CSV logs. It plots VRMSE versus k for
    selected epochs, plots the first and final epoch curves, and makes the parity
    plot from the test data produced by test.py.


Citation:
    ChatGPT was used for general code troubleshooting, generation of helper functions.