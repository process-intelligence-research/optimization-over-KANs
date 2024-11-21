# optimization-over-KANs
This repository contains the [Pyomo](https://github.com/Pyomo/pyomo) files describing the proposed Mixed-Integer Nonlinear Programming formulation in the paper **`Deterministic Global Optimization over trained Kolmogorov Arnold Networks`** (TODO: Add the link to preprint once submitted). <br>
In addition, the repository also contain Python scripts to train multi-layer perceptrons (MLP) using Tensorflow and then optimizing over the trained MLPs using [OMLT](https://github.com/cog-imperial/OMLT). <br>
Effectively, this repository contains all the files needed to reproduce the results in the paper:<br>
(TODO: Add a BibTeX reference to the pre-print) <br>
If you use the formulation from this paper, please consider citing it as described above. <br>

## Details about the files in the repository
The description of different files in this repository is provided below to help reproduce the results from the paper:<br>
1. `requirements.txt` - contains all the packages (including versions) required to run the files in this repository <br>
2. `generate_dataset.py` - Generates a tensor containing all the inputs and outputs for a given optimization test function as defined by the user. The test functions considered for the paper are Peaks and Rosenbrock functions. <br>
3. `export_KAN.py` - Reads a trained KAN object and exports all relevant information about the trained KAN including bounds on intermediate variables generated via simple feasibility-based bounds tightening as described in the paper in a JSON file. <br>
4. `train_KAN.ipynb` - Jupyter notebook to train a KAN based on the generated data and export all the relevant information. <br>
5. `Pyomo_KAN_neuron.py` - A Python class to model a neuron in a Kolmogorov Arnold Network. It uses the block functionality of Pyomo. <br>
6. `Pyomo_KAN_layer.py` - A Python class to model a layer with all neurons in a Kolmogorov Arnold Network. It uses the block functionality of Pyomo. <br>
7. `create_KAN.py` - A Python script to connect all layers, inputs, outputs and corresponding scaling strategies based on the JSON file for the trained KAN. <br>
8. `KAN_formulation_options.json` - A JSON file comprising all possible formulation options described in the paper. Activate/deactivate as necessary.
9. `solve_KAN.py` - A Python script to solve an instantiated instance of a trained KAN from `create_KAN.py` using the formulation and solver of your choice. To do so run this Python file from command line. An example is provided below: <br>
`python solve_KAN.py <trained_KAN.json> KAN_formulation_options.json <solver>` <br>
10. `train_tf_mlp.py` - A Python script to train a MLP using Tensorflow. The training and test data along with the scaling strategy (in a JSON file) needs to be provided by the user. Look at the data provided in the Electronic Supplementary Information for the paper for additional details. 
11. `mlp_opt.py` - A Python script to import the tensorflow object into OMLT and optimize over it.

Please contact [Tanuj Karia](mailto:t.karia@tudelft.nl) if you need any help in running this repository.