import argparse
import json
from omlt import OffsetScaling, OmltBlock
from omlt.io.keras import load_keras_sequential
from omlt.neuralnet import ReluBigMFormulation, ReluPartitionFormulation, ReluComplementarityFormulation
import pyomo.environ as pyo
from tensorflow import keras
import os

# Set up argument parsing for command-line execution
parser = argparse.ArgumentParser(description="Load Keras model, scaler, and solver parameters.")
parser.add_argument('--keras_model', type=str, required=True, help="Path to the Keras model file.")
parser.add_argument('--scaler_file', type=str, required=True, help="Path to the scaler JSON file.")
parser.add_argument('--formulation', type=str, required=True, choices=['bigm', 'partition', 'complementarity'], 
                    help="Type of formulation to use: 'bigm', 'partition', or 'complementarity'.")
parser.add_argument('--solver', type=str, required=True, help="Solver to use (e.g., scip, gurobi, cplex).")
parser.add_argument('--num_inputs', type=int, required=True, help="Number of input variables.")
parser.add_argument('--input_lb', type=float, required=True, help="Lower bound of the input variables.")
parser.add_argument('--input_ub', type=float, required=True, help="Upper bound of the input variables.")
parser.add_argument('--time_limit', type=int, default=7200, help="Time limit for the solver in seconds (default: 7200).")

args = parser.parse_args()

# Extract the base name of the Keras model file (to name the log and results files)
keras_model_base = os.path.splitext(os.path.basename(args.keras_model))[0]

# Generate log file and JSON results file names
log_file_name = f"{keras_model_base}_{args.formulation}.log"
json_file_name = f"{keras_model_base}_{args.formulation}.json"

# Create a Pyomo concrete model
m = pyo.ConcreteModel()

# Add an OMLT surrogate block to the Pyomo model
m.surrogate = OmltBlock()

# Load the Keras model from the specified file
nn = keras.models.load_model(args.keras_model, compile=False)

# Load the scaler information from the JSON file
with open(args.scaler_file, 'r') as file:
    scaler_dict = json.load(file)

# Print the loaded scaler details for verification
print("Scaler details loaded:", scaler_dict)

# Create an OffsetScaling object for input-output normalization
scaler = OffsetScaling(
    offset_inputs={i: scaler_dict['mean_input'][0][i] for i in range(args.num_inputs)},
    factor_inputs={i: scaler_dict['std_input'][0][i] for i in range(args.num_inputs)},
    offset_outputs={i: scaler_dict['mean_label'][0][i] for i in range(1)},  # Assuming one output
    factor_outputs={i: scaler_dict['std_label'][0][i] for i in range(1)}
)

# Calculate scaled input bounds based on the given lower and upper bounds
input_lb = args.input_lb
input_ub = args.input_ub
scaled_lb = []
scaled_ub = []

# Compute scaled bounds for each input variable
for i in range(args.num_inputs):
    scaled_lb.append((input_lb - scaler_dict['mean_input'][0][i]) / scaler_dict['std_input'][0][i])
    scaled_ub.append((input_ub - scaler_dict['mean_input'][0][i]) / scaler_dict['std_input'][0][i])

scaled_input_bounds = {i: (scaled_lb[i], scaled_ub[i]) for i in range(args.num_inputs)}

# Load the Keras model as an OMLT neural network with scaling
net = load_keras_sequential(nn, scaling_object=scaler, scaled_input_bounds=scaled_input_bounds)

# Choose the formulation type based on the user's input
if args.formulation == 'bigm':
    formulation = ReluBigMFormulation(net)
elif args.formulation == 'partition':
    formulation = ReluPartitionFormulation(net)
elif args.formulation == 'complementarity':
    formulation = ReluComplementarityFormulation(net)

# Build the surrogate model in Pyomo using the selected formulation
m.surrogate.build_formulation(formulation)

# Define the objective function (maximize/minimize the neural network output)
# Customize this function based on your specific use case
m.objective = pyo.Objective(expr=m.surrogate.outputs[0])

# Set up the solver with the specified time limit
solver = pyo.SolverFactory(args.solver)

# Solve the Pyomo model and log the results
status = solver.solve(m, tee=True, timelimit=args.time_limit, logfile=log_file_name)

# Check the solver's termination condition and print the status
if status.solver.termination_condition == pyo.TerminationCondition.optimal:
    print("Optimal solution found.")
else:
    print(f"Solver terminated with condition: {status.solver.termination_condition}")

# Create a results dictionary to store unscaled inputs and output values
results_dict = {}

# Store the unscaled input values and output (assuming one output variable)
for i in range(args.num_inputs):
    results_dict[f'x{i+1}'] = m.surrogate.inputs[i].value  # Adjust according to input storage
results_dict['f*'] = m.surrogate.outputs[0].value  # Output value

# Save the results to a JSON file
with open(json_file_name, 'w') as f:
    json.dump(results_dict, f, default=str, indent=4)

print(f"Results saved to {json_file_name}.")