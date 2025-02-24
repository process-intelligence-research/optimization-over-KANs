"""Module to create a Pyomo model by reading a trained KAN model object.
   This module imports the layer class and completes all the connections
   from the input to the output layer.
   The bounds on unscaled input and output variables should be defined here."""
import json
import argparse
import pyomo.environ as pyo
from pyomo_kan_layer import LayerBlockRule

# Function to parse command-line arguments
def parse_args():
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments with the path to the JSON data file.
    """
    parser = argparse.ArgumentParser(
        description='Create a Pyomo model using the provided JSON data file.')
    parser.add_argument(
        'json_file', type=str, help='Path to the JSON data file.')
    parser.add_argument(
        'options_file', type=str, help='Path to the options JSON file.')
    return parser.parse_args()

# Function to load model data from a JSON file
def load_model_data(json_file):
    """
    Reads JSON data from a file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data as a Python dictionary.
    """
    with open(json_file, 'r', encoding="utf-8") as f:
        return json.load(f)

# Main function to create the Pyomo model
def main(json_file, options_file):
    """
    Creates a Pyomo model based on the JSON data and options.

    Args:
        json_file (str): Path to the JSON file containing model data.
        options_file (str): Path to the JSON file containing configuration options.

    Returns:
        pyo.ConcreteModel: The constructed Pyomo model.
    """
    # Load model and options data
    model_data = load_model_data(json_file)
    options = load_model_data(options_file)

    # Unpack necessary data from the JSON file
    mean_input_dict = model_data['mean_input_dict']  # Mean values for input scaling
    std_input_dict = model_data['std_input_dict']  # Standard deviations for input scaling
    mean_output_dict = model_data['mean_output_dict']  # Mean values for output scaling
    std_output_dict = model_data['std_output_dict']  # Standard deviations for output scaling
    num_inputs_dict = model_data['num_inputs_dict']  # Number of inputs per layer
    num_outputs_dict = model_data['num_outputs_dict']  # Number of outputs per layer
    depth = model_data['depth']  # Total number of layers in the model

    # Define layers in the network (excluding input layer)
    layers = list(range(depth - 1))

    # Create a Pyomo model
    model = pyo.ConcreteModel()
    model.layers = pyo.Set(initialize=layers)

    # Define a block rule for each layer
    def layer_block_rule(block, nl):
        """
        Rule for initializing a layer block.

        Args:
            block (pyo.Block): The Pyomo block for the layer.
            nl (int): The index of the layer.
        """
        LayerBlockRule(block, nl, model_data, options)

    # Initialize blocks for all layers
    model.layer_block = pyo.Block(model.layers, rule=layer_block_rule)

    # Define unscaled input and output variables
    model.unscaled_inputs = pyo.Var(
        model.layer_block[0].inputs, bounds=(-2.048, 2.048))  # Input bounds
    model.unscaled_outputs = pyo.Var(
        model.layer_block[depth - 2].outputs)  # Outputs from the last layer

    # Add input constraints for the first layer
    for i in range(num_inputs_dict["0"]):
        # Scale the input variables
        model.layer_block[0].constraints.add(
            model.layer_block[0].neurons[0, i].input * std_input_dict[f"{i}"] ==
            (model.unscaled_inputs[i] - mean_input_dict[f"{i}"])
        )
    for i in range(num_inputs_dict["0"]):
        for j in range(1, num_outputs_dict["0"]):
            # Ensure all neurons in the first layer share the same input
            model.layer_block[0].constraints.add(
                model.layer_block[0].neurons[j, i].input == model.layer_block[0].neurons[0, i].input
            )

    # Add constraints to connect the layers
    for l in range(1, depth - 1):
        for i in range(num_inputs_dict[str(l)]):
            for j in range(num_outputs_dict[str(l)]):
                # Set the input of the current layer neurons to the output of the previous layer
                model.layer_block[l].constraints.add(
                    model.layer_block[l].neurons[j, i].input ==
                    model.layer_block[l - 1].layer_outputs[i]
                )

    # Add output constraints for the last layer
    for i in range(num_outputs_dict[str(depth - 2)]):
        # Scale the outputs
        model.layer_block[depth - 2].constraints.add(
            model.layer_block[depth - 2].layer_outputs[i] * std_output_dict[f"{i}"] ==
            (model.unscaled_outputs[i] - mean_output_dict[f"{i}"])
        )

    # Define the objective function (minimize the first unscaled output)
    model.objective = pyo.Objective(expr=model.unscaled_outputs[0], sense=pyo.minimize)

    return model

# Entry point for the script
if __name__ == "__main__":
    args = parse_args()
    kan_model = main(args.json_file, args.options_file)
    print("Model creation complete.")
