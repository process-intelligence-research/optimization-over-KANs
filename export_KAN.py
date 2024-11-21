import numpy as np
import math
import json

def extend_grid(grid, k_extend=0):
    """
    Extends the grid by padding `k_extend` values to the left and right.

    Parameters:
    -----------
    grid : np.ndarray
        Input grid with shape (batch, grid).
    k_extend : int
        Number of values to pad to the left and right. Default: 0.

    Returns:
    --------
    np.ndarray
        Extended grid.
    """
    # Calculate step size
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    # Pad values to the left and right
    for _ in range(k_extend):
        grid = np.concatenate([grid[:, [0]] - h, grid], axis=1)
        grid = np.concatenate([grid, grid[:, [-1]] + h], axis=1)
    return grid


def tuple_to_str(t):
    """
    Converts a tuple to a comma-separated string.

    Parameters:
    -----------
    t : tuple
        Input tuple to convert.

    Returns:
    --------
    str
        Comma-separated string representation of the tuple.
    """
    return ','.join(map(str, t))


def get_model_data(model, dataset):
    """
    Extracts model data and associated properties for serialization.

    Parameters:
    -----------
    model : object
        Model object containing network properties and methods.
    dataset : dict
        Dictionary containing training inputs, labels, and normalization parameters.

    Returns:
    --------
    dict
        Dictionary containing extracted model data and metadata.
    """
    # Initialize dimensions and constants
    width = model.width
    depth = len(width)
    degree = 3
    coefs_size = model.act_fun[0].coef[0].shape[0]
    knots_size = coefs_size + degree + 1

    # Initialize dictionaries for various attributes
    num_inputs_dict = {}
    num_outputs_dict = {}
    for l in range(depth - 1):
        num_inputs_dict[str(l)] = width[l]
        num_outputs_dict[str(l)] = width[l + 1]

    coefs_dict = {}
    grids_dict = {}
    scale_sp_dict = {}
    scale_base_dict = {}
    biases_dict = {}
    input_lb_dict = {}
    input_ub_dict = {}
    output_lb_dict = {}
    output_ub_dict = {}
    knots_dict = {}

    # Run model on training data to initialize weights and attributes
    model(dataset['train_input'])
    mean_input = dataset['mean_input'].detach().numpy()
    std_input = dataset['std_input'].detach().numpy()
    mean_label = dataset['mean_label'].detach().numpy()
    std_label = dataset['std_label'].detach().numpy()

    # Normalize input and output means and standard deviations
    mean_input_dict = {str(i): mean_input[0][i] for i in range(mean_input.shape[1])}
    std_input_dict = {str(i): std_input[0][i] for i in range(std_input.shape[1])}
    mean_output_dict = {str(i): mean_label[0][i] for i in range(mean_label.shape[1])}
    std_output_dict = {str(i): std_label[0][i] for i in range(std_label.shape[1])}

    # Process layers of the model
    for l in range(depth - 1):
        num_inputs = width[l]
        num_outputs = width[l + 1]
        for j in range(num_outputs):
            for i in range(num_inputs):
                index = num_inputs * j + i
                # Process grid and coefficients
                grid = model.act_fun[l].grid[index]
                input_lb_dict[tuple_to_str((l, j, i))] = grid[0].detach().item()
                input_ub_dict[tuple_to_str((l, j, i))] = grid[-1].detach().item()
                grid = grid[None, :].detach().numpy()
                grids_dict[tuple_to_str((l, j, i))] = grid.tolist()  # JSON serializable
                extended_grid = extend_grid(grid, 3)
                for k in range(extended_grid.shape[1]):
                    knots_dict[tuple_to_str((l, j, i, k))] = extended_grid[0][k]
                coef = model.act_fun[l].coef[index].detach().numpy()
                for idx, c in enumerate(coef):
                    coefs_dict[tuple_to_str((l, j, i, idx))] = c
                # Scale parameters
                scale_base = model.act_fun[l].scale_base[index].detach().item()
                scale_base_dict[tuple_to_str((l, j, i))] = scale_base
                scale_sp = model.act_fun[l].scale_sp[index].detach().item()
                scale_sp_dict[tuple_to_str((l, j, i))] = scale_sp
                # Node ranges
                node_range = model.get_range(l, i, j)
                output_lb_dict[tuple_to_str((l, j, i))] = node_range[2].detach().item()
                output_ub_dict[tuple_to_str((l, j, i))] = node_range[3].detach().item()
            # Biases
            bias = model.biases[l].weight.data[0][j].detach().item()
            biases_dict[tuple_to_str((l, j))] = bias

    # Calculate upper bounds for the SiLU activation
    silu_output_ub_dict = {}
    for l in range(depth - 1):
        for j in range(num_outputs_dict[str(l)]):
            for i in range(num_inputs_dict[str(l)]):
                ub = input_ub_dict[tuple_to_str((l, j, i))]
                silu_output_ub_dict[tuple_to_str((l, j, i))] = ub / (1 + math.exp(-ub))

    # Calculate layer-level bounds
    layer_output_lb_dict = {}
    layer_output_ub_dict = {}
    for l in range(depth - 1):
        for j in range(num_outputs_dict[str(l)]):
            lb_sum = sum(output_lb_dict[f"{l},{j},{i}"] for i in range(num_inputs_dict[str(l)]))
            ub_sum = sum(output_ub_dict[f"{l},{j},{i}"] for i in range(num_inputs_dict[str(l)]))
            layer_output_lb_dict[f"{l},{j}"] = lb_sum + biases_dict[f"{l},{j}"]
            layer_output_ub_dict[f"{l},{j}"] = ub_sum + biases_dict[f"{l},{j}"]

    # Compile data dictionary
    data = {
        'width': width,
        'depth': depth,
        'degree': degree,
        'coefs_size': coefs_size,
        'knots_size': knots_size,
        'num_inputs_dict': num_inputs_dict,
        'num_outputs_dict': num_outputs_dict,
        'coefs_dict': coefs_dict,
        'knots_dict': knots_dict,
        'scale_sp_dict': scale_sp_dict,
        'scale_base_dict': scale_base_dict,
        'biases_dict': biases_dict,
        'input_lb_dict': input_lb_dict,
        'input_ub_dict': input_ub_dict,
        'output_lb_dict': output_lb_dict,
        'output_ub_dict': output_ub_dict,
        'layer_output_lb_dict': layer_output_lb_dict,
        'layer_output_ub_dict': layer_output_ub_dict,
        'silu_output_ub_dict': silu_output_ub_dict,
        'mean_input_dict': mean_input_dict,
        'std_input_dict': std_input_dict,
        'mean_output_dict': mean_output_dict,
        'std_output_dict': std_output_dict
    }

    return data


def export_to_json(data, filename):
    """
    Exports data to a JSON file.

    Parameters:
    -----------
    data : dict
        Data to export.
    filename : str
        Name of the output JSON file.

    Returns:
    --------
    None
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)