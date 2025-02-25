"""Module to train a KAN model.
   For additional information about this please refer to:
   https://kindxiaoming.github.io/pykan/  """
from kan import *
from util.generate_dataset import create_custom_dataset
from util.export_kan import get_model_data, export_to_json

# Creating the functions investigated for data generation
def peaks_function(x):
    """
    Computes the Peaks function for a given input tensor.
    
    The Peaks function is a commonly used 2D test function in optimization and 
    machine learning, featuring multiple local maxima and minima.

    Args:
        x (torch.Tensor): A tensor of shape (N, 2), where each row represents 
                          a 2D point (x1, x2).

    Returns:
        torch.Tensor: A tensor of shape (N, 1) containing the function values.
    """
    return (
        3 * (1 - x[:, 0]) ** 2 * torch.exp(-x[:, 0]**2 - (x[:, 1] + 1)**2) -
        10 * (x[:, 0] / 5 - x[:, 0]**3 - x[:, 1]**5) * torch.exp(-x[:, 0]**2 - x[:, 1]**2) -
        torch.exp(-(x[:, 0] + 1)**2 - x[:, 1]**2) / 3
    ).unsqueeze(1)

def rosenbrock_3(x):
    """
    Computes the 3-dimensional Rosenbrock function.

    The Rosenbrock function is a common test problem for optimization algorithms, 
    characterized by a narrow, curved valley leading to the global minimum.

    Args:
        x (torch.Tensor): A tensor of shape (N, 3), where each row represents 
                          a 3D point (x1, x2, x3).

    Returns:
        torch.Tensor: A tensor of shape (N, 1) containing the function values.
    """
    return (
        100 * torch.pow(x[:, 1] - torch.pow(x[:, 0], 2), 2) + torch.pow(1 - x[:, 0], 2) +
        100 * torch.pow(x[:, 2] - torch.pow(x[:, 1], 2), 2) + torch.pow(1 - x[:, 1], 2)
    ).unsqueeze(1)

def rosenbrock_5(x):
    """
    Computes the 5-dimensional Rosenbrock function.

    This function extends the classic Rosenbrock function to five dimensions.

    Args:
        x (torch.Tensor): A tensor of shape (N, 5), where each row represents 
                          a 5D point (x1, x2, x3, x4, x5).

    Returns:
        torch.Tensor: A tensor of shape (N, 1) containing the function values.
    """
    return (
        100 * torch.pow(x[:, 1] - torch.pow(x[:, 0], 2), 2) + torch.pow(1 - x[:, 0], 2) +
        100 * torch.pow(x[:, 2] - torch.pow(x[:, 1], 2), 2) + torch.pow(1 - x[:, 1], 2) +
        100 * torch.pow(x[:, 3] - torch.pow(x[:, 2], 2), 2) + torch.pow(1 - x[:, 2], 2) +
        100 * torch.pow(x[:, 4] - torch.pow(x[:, 3], 2), 2) + torch.pow(1 - x[:, 3], 2)
    ).unsqueeze(1)

def rosenbrock_10(x):
    """
    Computes the 10-dimensional Rosenbrock function.

    The 10D Rosenbrock function is a more complex version of the original function, 
    often used to test optimization algorithms in high-dimensional spaces.

    Args:
        x (torch.Tensor): A tensor of shape (N, 10), where each row represents 
                          a 10D point (x1, x2, ..., x10).

    Returns:
        torch.Tensor: A tensor of shape (N, 1) containing the function values.
    """
    return (
        100 * torch.pow(x[:, 1] - torch.pow(x[:, 0], 2), 2) + torch.pow(1 - x[:, 0], 2) +
        100 * torch.pow(x[:, 2] - torch.pow(x[:, 1], 2), 2) + torch.pow(1 - x[:, 1], 2) +
        100 * torch.pow(x[:, 3] - torch.pow(x[:, 2], 2), 2) + torch.pow(1 - x[:, 2], 2) +
        100 * torch.pow(x[:, 4] - torch.pow(x[:, 3], 2), 2) + torch.pow(1 - x[:, 3], 2) +
        100 * torch.pow(x[:, 5] - torch.pow(x[:, 4], 2), 2) + torch.pow(1 - x[:, 4], 2) +
        100 * torch.pow(x[:, 6] - torch.pow(x[:, 5], 2), 2) + torch.pow(1 - x[:, 5], 2) +
        100 * torch.pow(x[:, 7] - torch.pow(x[:, 6], 2), 2) + torch.pow(1 - x[:, 6], 2) +
        100 * torch.pow(x[:, 8] - torch.pow(x[:, 7], 2), 2) + torch.pow(1 - x[:, 7], 2) +
        100 * torch.pow(x[:, 9] - torch.pow(x[:, 8], 2), 2) + torch.pow(1 - x[:, 8], 2)
    ).unsqueeze(1)

# Update the function and relevant function arguments here
dataset = create_custom_dataset(
    peaks_function,
    n_var=2,
    ranges=(-3,3),
    train_num=800,
    test_num=200,
    normalize_input=True,
    normalize_label=True)

# Initializing the list of elements of grid-size for iterative refinement of grid-size for a KAN
grids = np.array([3,4,5,6])
train_losses = [] # Empty list to store train loss history
test_losses = [] # Empty list to store test loss history
STEPS = 125 # Number of iterations to train the KAN for
K = 3 # Degree of B-spline activations in KAN
WIDTH = [2, 2, 1] # Defining KAN architecture
# Step-wise training of KAN with gradual increase in grid-size

# Begin training loop
for i in range(grids.shape[0]):
    if i == 0:
        # Initialising the KAN with the first element of the grid
        model = KAN(width=WIDTH,
                    grid=grids[i],
                    k=K,
                    seed=0) # Fixing the seed to reproduce the models
    if i != 0:
        # For all other elements in grids
        model = KAN(
            width=WIDTH,
            grid=grids[i],
            k=K
            ).initialize_from_another_model(model, dataset['train_input']) # Refining grid-size
    results = model.train(dataset, opt="LBFGS", steps=STEPS)
    train_losses += results['train_loss']
    test_losses += results['test_loss']

# Exporting trained KAN as a JSON file
data = get_model_data(model, dataset)
# Name of the JSON file based on no. of hidden layers (H),
# no. of neurons (N), and no. of grid points (G)
export_to_json(data,'Peaks_H1_N2_G6.json')
