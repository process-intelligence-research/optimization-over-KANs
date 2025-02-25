"""This module generates the dataset given a symbolic function.
   Examples of symbolic functions include peaks function, rosenbrock function, etc.
   This module also returns the mean and standard deviation of the data generated."""

import numpy as np
import torch

def create_custom_dataset(
    f,
    n_var=2,
    ranges=None,
    train_num=1000,
    test_num=1000,
    normalize_input=False,
    normalize_label=False,
    device='cpu',
    seed=0
):
    '''
    Create dataset
    
    Args:
    -----
        f : function
            The symbolic formula used to create the synthetic dataset
        n_var : int
            Number of input variables. Default: 2.
        ranges : list or np.array; shape (2,) or (n_var, 2)
            The range of input variables. Default: [-1, 1].
        train_num : int
            The number of training samples. Default: 1000.
        test_num : int
            The number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            Device to place the dataset on. Default: 'cpu'.
        seed : int
            Random seed. Default: 0.
        
    Returns:
    --------
        dataset : dict
            A dictionary containing train/test inputs/labels:
            - dataset['train_input']
            - dataset['train_label']
            - dataset['test_input']
            - dataset['test_label']
            - dataset['mean_input'] (if normalized)
            - dataset['std_input'] (if normalized)
            - dataset['mean_label'] (if normalized)
            - dataset['std_label'] (if normalized)
         
    Example:
    --------
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> dataset = create_dataset_custom(f, n_var=2, train_num=100)
    >>> dataset['train_input'].shape
    torch.Size([100, 2])
    '''
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Normalize ranges
    if range is None:
        ranges = [-1,1] # Initializing default value for the range
    if len(np.array(ranges).shape) == 1:
        ranges = np.array(ranges * n_var).reshape(n_var, 2)
    else:
        ranges = np.array(ranges)
    # Initialize inputs
    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    # Generate data within the specified ranges
    for i in range(n_var):
        train_input[:, i] = torch.rand(train_num) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]
        test_input[:, i] = torch.rand(test_num) * (ranges[i, 1] - ranges[i, 0]) + ranges[i, 0]
    # Compute labels
    train_label = f(train_input)
    test_label = f(test_input)
    # Normalization helper
    def normalize(data, mean, std):
        return (data - mean) / std

    # Normalize inputs if specified
    mean_input, std_input, mean_label, std_label = None, None, None, None
    if normalize_input:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)
    # Normalize labels if specified
    if normalize_label:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    # Package dataset
    dataset = {
        'train_input': train_input.to(device),
        'test_input': test_input.to(device),
        'train_label': train_label.to(device),
        'test_label': test_label.to(device)
    }
    if normalize_input:
        dataset['mean_input'] = mean_input
        dataset['std_input'] = std_input
    if normalize_label:
        dataset['mean_label'] = mean_label
        dataset['std_label'] = std_label

    return dataset
