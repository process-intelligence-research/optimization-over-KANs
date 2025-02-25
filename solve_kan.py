"""This module reads a KAN model in Pyomo using the create_kan module
   and invokes the solve statement in Pyomo to solve them.
   The main purpose of this module is to allow the solution of KAN models
   from the command line using the json file describing the KAN model, 
   the json file containing the formulation options, and the solver name as arguments."""
import argparse
import json
import pyomo.environ as pyo
from src.create_kan import main

# Function to parse command-line arguments
def parse_args():
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: 
        Parsed arguments containing paths to the JSON data file, options file, and the solver name.
    """
    parser = argparse.ArgumentParser(
        description='Solve the Pyomo model using the specified solver.')
    parser.add_argument('json_file', type=str, help='Path to the JSON data file.')
    parser.add_argument('options_file', type=str, help='Path to the options JSON file.')
    parser.add_argument(
        'solver', type=str, help='Solver to use for solving the Pyomo model (e.g., "ipopt").')
    return parser.parse_args()

# Main function to solve the Pyomo model
def solve_model(json_file, options_file, solver_name):
    """
    Solves the Pyomo model using the specified solver and saves results to a JSON file.

    Args:
        json_file (str): Path to the JSON file containing model data.
        options_file (str): Path to the JSON file containing solver options or configurations.
        solver_name (str): Name of the solver to use.

    Raises:
        ValueError: If the specified solver is not available.

    Outputs:
        - A log file containing the solver output.
        - A JSON file containing the solution values.
    """
    # Create the Pyomo model using the data and options
    model = main(json_file, options_file)
    # Initialize the solver
    solver = pyo.SolverFactory(solver_name)
    # Check if the solver is available
    if not solver.available():
        raise ValueError(f"Solver {solver_name} is not available.")
    # Define output file names
    log_file = json_file.replace('.json', '.log')  # Solver log file
    results_file = json_file.replace('.json', '_results.json')  # Results file
    # Solve the model
    result = solver.solve(model, tee=True, timelimit=7200, logfile=log_file)
    # Check the solver's termination condition
    if result.solver.termination_condition == pyo.TerminationCondition.optimal:
        print("Optimal solution found.")
    else:
        print(f"Solver terminated with condition: {result.solver.termination_condition}")
    # Extract results and store them in a dictionary
    results_dict = {}
    # Retrieve values for specific unscaled inputs and outputs
    results_dict['f*'] = model.unscaled_outputs[0].value
    results_dict['x1'] = model.unscaled_inputs[0].value
    results_dict['x2'] = model.unscaled_inputs[1].value
    # Uncomment below lines if additional variables need to be extracted
	# results_dict['x3'] = model.unscaled_inputs[2].value
    # results_dict['x4'] = model.unscaled_inputs[3].value
    # results_dict['x5'] = model.unscaled_inputs[4].value
    # results_dict['x6'] = model.unscaled_inputs[5].value
    # results_dict['x7'] = model.unscaled_inputs[6].value
    # results_dict['x8'] = model.unscaled_inputs[7].value
    # results_dict['x9'] = model.unscaled_inputs[8].value
    # results_dict['x10'] = model.unscaled_inputs[9].value
    # Write the results to a JSON file
    with open(results_file, 'w', encoding="utf-8") as f:
        json.dump(results_dict, f, default=str, indent=4)
    print(f"Results saved to {results_file}")

# Entry point for the script
if __name__ == "__main__":
    args = parse_args()
    solve_model(args.json_file, args.options_file, args.solver)
