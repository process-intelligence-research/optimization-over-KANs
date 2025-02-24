import pyomo.environ as pyo
import json
from Pyomo_KAN_neuron import NeuronBlockRule

class LayerBlockRule:
    """
    A class to define a Pyomo block for a single layer in a neural network, 
    with support for initializing neuron blocks and constraints.

    Attributes:
        block (pyo.Block): The Pyomo block representing the layer.
        nl (int): The layer index.
        model_data (dict): Data dictionary containing parameters like input/output sizes and biases.
        options (dict): Configuration options for neuron block initialization.
    """
    def __init__(self, block, nl, model_data, options):
        """
        Initialize the layer block with the necessary parameters and constraints.

        Args:
            block (pyo.Block): The Pyomo block to initialize for the layer.
            nl (int): The layer index.
            model_data (dict): Data dictionary with network and layer-specific parameters.
            options (dict): Configuration options for neuron and layer block initialization.
        """
        self.block = block
        self.nl = nl
        self.model_data = model_data
        self.options = options
        self.initialize_block()

    def initialize_block(self):
        """
        Initialize the Pyomo block for the layer with neurons, outputs, and constraints.
        """
        block = self.block
        nl = self.nl
        model_data = self.model_data

        # Retrieve parameters for the layer
        num_inputs = model_data['num_inputs_dict'][str(nl)]  # Number of input nodes to the layer
        num_outputs = model_data['num_outputs_dict'][str(nl)]  # Number of output nodes in the layer
        bias_dict = model_data['biases_dict']  # Biases for each neuron
        layer_outputs_lb_dict = model_data['layer_output_lb_dict']  # Lower bounds for layer outputs
        layer_outputs_ub_dict = model_data['layer_output_ub_dict']  # Upper bounds for layer outputs

        # Define sets for the layer's input and output nodes
        inputs = list(range(num_inputs))  # Indices of input nodes
        outputs = list(range(num_outputs))  # Indices of output nodes
        block.inputs = pyo.Set(initialize=inputs)  # Set of input indices
        block.outputs = pyo.Set(initialize=outputs)  # Set of output indices

        # Function to initialize neuron blocks for each neuron
        def neuron_block_init(neuron_block, no, ni):
            """
            Initialize the Pyomo block for an individual neuron in the layer.

            Args:
                neuron_block (pyo.Block): The Pyomo block for the neuron.
                no (int): The output node index in the layer.
                ni (int): The input node index to the neuron.
            """
            NeuronBlockRule(neuron_block, nl, no, ni, model_data, self.options)

        # Create a Pyomo block for each neuron in the layer
        block.neurons = pyo.Block(block.outputs, block.inputs, rule=neuron_block_init)

        # Define variables for layer outputs
        block.layer_outputs = pyo.Var(block.outputs, within=pyo.Reals)
        for no in outputs:
            # Set bounds on the layer outputs
            block.layer_outputs[no].setlb(layer_outputs_lb_dict[f"{nl},{no}"])
            block.layer_outputs[no].setub(layer_outputs_ub_dict[f"{nl},{no}"])

        # Add constraints to relate neuron outputs to the layer output
        block.constraints = pyo.ConstraintList()
        for j in range(num_outputs):
            # Each layer output is the sum of outputs from all neurons in the layer, plus a bias
            block.constraints.add(
                block.layer_outputs[j] == sum(block.neurons[j, i].neuron_output for i in range(num_inputs)) + bias_dict[f"{nl},{j}"]
            )
