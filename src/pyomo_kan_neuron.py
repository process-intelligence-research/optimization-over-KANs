""" Module modelling a neuron in a Kolmogorov Arnold Network 
    in Pyomo to embed in optimisation models"""

import pyomo.environ as pyo

class NeuronBlockRule:
    """
    A class to define a Pyomo block representing a neuron in a neural network,
    including constraints for spline-based activations and SiLU activation functions.

    Attributes:
        block (pyo.Block): The Pyomo block for the neuron.
        nl (int): Layer index of the neuron.
        no (int): Output node index within the layer.
        ni (int): Input node index within the layer.
        model_data (dict): Data dictionary containing model parameters and constraints.
        options (dict): Configuration options for formulating the block.
    """
    def __init__(self, block, nl, no, ni, model_data, options):
        """
        Initialize the neuron block with necessary parameters and constraints.

        Args:
            block (pyo.Block): The Pyomo block to initialize.
            nl (int): Layer index.
            no (int): Output node index within the layer.
            ni (int): Input node index within the layer.
            model_data (dict): Data containing model parameters (e.g., coefficients, bounds).
            options (dict): Formulation options for reformulations, cuts, and sparsity.
        """
        self.block = block
        self.nl = nl
        self.no = no
        self.ni = ni
        self.model_data = model_data
        self.options = options
        self.initialize_block()

    def apply_big_m_reformulation(self):
        """
        Apply Big-M reformulation to enforce constraints 
        on the neuron's spline input-output relationship.
        """
        block = self.block
        nl = self.nl
        no = self.no
        ni = self.ni
        coefs_size = self.model_data['coefs_size']
        degree = self.model_data['degree']
        knots_dict = self.model_data['knots_dict']

        for i in range(degree + coefs_size):
            block.constraints.add(
                block.input >= (knots_dict[f"{nl},{no},{ni},{i}"] - knots_dict[f"{nl},{no},{ni},0"])
                * block.N[i, 0] + knots_dict[f"{nl},{no},{ni},0"])
            block.constraints.add(
                block.input <= (knots_dict[f"{nl},{no},{ni},{i + 1}"]
                                - knots_dict[f"{nl},{no},{ni},{degree + coefs_size}"])
                * block.N[i, 0] + knots_dict[f"{nl},{no},{ni},{degree + coefs_size}"])

    def apply_convex_hull_reformulation(self):
        """
        Apply Convex Hull reformulation to enforce constraints on the spline representation.
        This provides a tighter representation compared to Big-M.
        """
        block = self.block
        nl = self.nl
        no = self.no
        ni = self.ni
        coefs_size = self.model_data['coefs_size']
        knots_size = self.model_data['knots_size']
        degree = self.model_data['degree']
        knots_dict = self.model_data['knots_dict']

        block.indices = pyo.RangeSet(0, knots_size - 1)
        block.z = pyo.Var(block.indices, domain=pyo.Reals)

        for i in range(degree + coefs_size):
            block.constraints.add(
                block.z[i] <= knots_dict[f"{nl},{no},{ni},{i + 1}"] * block.N[i, 0])
            block.constraints.add(
                block.z[i] >= knots_dict[f"{nl},{no},{ni},{i}"] * block.N[i, 0])

        block.constraints.add(sum(block.z[i] for i in block.indices) == block.input)

    def apply_exploit_sparsity(self):
        """
        Exploit sparsity in higher-degree basis functions by fixing irrelevant variables to zero.
        """
        block = self.block
        degree = self.model_data['degree']
        coefs_size = self.model_data['coefs_size']

        for dd in range(degree + 1):
            for d in range(degree - dd):
                block.N[d, dd].fix(0)
                block.N[coefs_size + d, dd].fix(0)

    def add_redundant_cuts(self):
        """
        Add redundant cuts to strengthen the model by linking basis function variables.
        """
        block = self.block
        degree = self.model_data['degree']
        coefs_size = self.model_data['coefs_size']

        for j in range(1, degree + 1):
            for i in range(coefs_size + degree - j):
                block.constraints.add(block.N[i, j] <= block.N[i, j - 1] + block.N[i + 1, j - 1])

    def add_local_support_cuts(self):
        """
        Add local support cuts to improve the constraint system for B-spline basis functions.
        """
        block = self.block
        degree = self.model_data['degree']
        coefs_size = self.model_data['coefs_size']

        for dd in range(degree):
            for d in range(dd + 1, degree + 1):
                for i in range(coefs_size):
                    block.constraints.add(
                        block.N[i, d] <= sum(block.N[j, dd] for j in range(i, i + d + 1 - dd)))

    def strengthen_silu(self):
        """
        Add strengthened McCormick envelopes for the SiLU activation function.
        """
        block = self.block
        nl = self.nl
        no = self.no
        ni = self.ni

        input_lb_dict = self.model_data['input_lb_dict']
        input_ub_dict = self.model_data['input_ub_dict']

        x_lb = max(input_lb_dict[f"{nl},{no},{ni}"], -709.77)
        x_ub = min(input_ub_dict[f"{nl},{no},{ni}"], 709.77)
        sigmoid_lb = 1 / (1 + pyo.exp(-x_lb))
        sigmoid_ub = 1 / (1 + pyo.exp(-x_ub))

        block.sigmoid = pyo.Var(bounds=(sigmoid_lb, sigmoid_ub))
        block.constraints.add(block.sigmoid == block.sigmoid_output)

        block.constraints.add(
            block.silu_output <= x_ub * block.sigmoid
            + block.input * sigmoid_lb - x_ub * sigmoid_lb)
        block.constraints.add(
            block.silu_output <= block.input * sigmoid_ub
            + x_lb * block.sigmoid - x_lb * sigmoid_ub)
        block.constraints.add(
            block.silu_output >= x_lb * block.sigmoid
            + block.input * sigmoid_lb - x_lb * sigmoid_lb)
        block.constraints.add(
            block.silu_output >= x_ub * block.sigmoid
            + block.input * sigmoid_ub - x_ub * sigmoid_ub)

    def initialize_block(self):
        """
        Initialize the Pyomo block with variables, parameters,
        and constraints based on the specified reformulation and options.
        """
        block = self.block
        nl, no, ni = self.nl, self.no, self.ni
        model_data = self.model_data
        options = self.options

        # Retrieve model parameters
        coefs_size = model_data['coefs_size']
        knots_size = model_data['knots_size']
        degree = model_data['degree']

        # Block parameters
        block.coefs_size = pyo.Param(initialize=coefs_size)
        block.knots_size = pyo.Param(initialize=knots_size)
        block.degree = pyo.Param(initialize=degree)

        # Define indices
        indices = [(i, j) for j in range(degree + 1) for i in range(coefs_size + degree - j + 1)]
        continuous_indices = [(i, j) for j in range(1, degree + 1)
                              for i in range(coefs_size + degree - j + 1)]

        block.overall_indices = pyo.Set(initialize=indices)
        block.continuous_indices = pyo.Set(initialize=continuous_indices)
        block.binary_indices = block.overall_indices - block.continuous_indices

        # Define variables
        block.N = pyo.Var(block.overall_indices, initialize=0)
        for i in block.binary_indices:
            block.N[i].domain = pyo.Binary
        for i in block.continuous_indices:
            block.N[i].domain = pyo.NonNegativeReals

        block.input = pyo.Var(bounds=
                              (model_data['input_lb_dict'][f"{nl},{no},{ni}"],
                               model_data['input_ub_dict'][f"{nl},{no},{ni}"]))
        block.spline_output = pyo.Var(domain=pyo.Reals)
        block.silu_output = pyo.Var(domain=pyo.Reals)
        block.silu_output.setlb(-2.78464596867598e-01)
        block.neuron_output = pyo.Var(domain=pyo.Reals)
        block.neuron_output.setlb(model_data['output_lb_dict'][f"{nl},{no},{ni}"])
        block.neuron_output.setub(model_data['output_ub_dict'][f"{nl},{no},{ni}"])
        block.sigmoid_output = pyo.Expression(expr=1 / (1 + pyo.exp(-block.input)))

        # Initialize constraint list
        block.constraints = pyo.ConstraintList()

        # Apply reformulation based on options
        reformulation = options.get('reformulation', 'Big-M')
        if reformulation == "Big-M":
            self.apply_big_m_reformulation()
        elif reformulation == "Convex Hull":
            self.apply_convex_hull_reformulation()

        # Add basis function constraints
        block.constraints.add(sum(block.N[i, 0] for i in range(degree + coefs_size)) == 1)

        for j in range(1, degree + 1):
            for i in range(coefs_size + degree - j):
                block.constraints.add(
                    block.N[i, j] == (
                        block.input - model_data['knots_dict'][f"{nl},{no},{ni},{i}"]) /
                    (model_data['knots_dict'][f"{nl},{no},{ni},{i + j}"] -
                     model_data['knots_dict'][f"{nl},{no},{ni},{i}"]) * block.N[i, j - 1] +
                    (model_data['knots_dict'][f"{nl},{no},{ni},{i + j + 1}"] - block.input) /
                    (model_data['knots_dict'][f"{nl},{no},{ni},{i + j + 1}"] -
                     model_data['knots_dict'][f"{nl},{no},{ni},{i + 1}"]) * block.N[i + 1, j - 1]
                )

        # Spline output
        block.constraints.add(
            sum(model_data['coefs_dict'][f"{nl},{no},{ni},{i}"] *
                block.N[i, degree] for i in range(coefs_size)) == block.spline_output
        )

        # SiLU activation function
        block.constraints.add(block.silu_output == block.input * block.sigmoid_output)

        # Neuron output
        block.constraints.add(
            block.neuron_output ==
            model_data['scale_base_dict'][f"{nl},{no},{ni}"] * block.silu_output +
            model_data['scale_sp_dict'][f"{nl},{no},{ni}"] * block.spline_output
        )

        # Partition of unity constraints
        for j in range(1, degree + 1):
            block.constraints.add(sum(block.N[i, j] for i in range(degree + coefs_size - j)) == 1)

        # Apply additional cuts and constraints based on options
        if options.get('exploit_sparsity', 0):
            self.apply_exploit_sparsity()
        if options.get('redundant_cuts', 0):
            self.add_redundant_cuts()
        if options.get('local_support', 0):
            self.add_local_support_cuts()
        if options.get('silu_mccormick', 0):
            self.strengthen_silu()
