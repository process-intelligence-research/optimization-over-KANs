This folder contains MLP models as keras files which is used to instantiate the Pyomo model for a trained MLP using [OMLT](https://github.com/cog-imperial/OMLT).
The models are provided for the four test functions considered in the paper listed below:
1. Peaks function
2. Rosenbrock function with 3 inputs (denoted as r3)
3. Rosenbrock function with 5 inputs (denoted as r5)
4. Rosenbrock function with 10 inputs (denoted as r10)

Each folder contains keras files whose names are appended as:
- \<function\>_mlp_relu_X_XX

For example, a file named as `r3_mlp_relu_2_64.keras` corresponds to a MLP with two hidden layers with sixty four neurons each in the hidden layer modelling the rosenbrock function with three inputs.
