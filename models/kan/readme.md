This folder contains KAN models as JSON files which is used to instantiate the Pyomo model for a trained KAN as described in the paper.
The models are provided for the four test functions considered in the paper listed below:
1. Peaks function
2. Rosenbrock function with 3 inputs (denoted as r3)
3. Rosenbrock function with 5 inputs (denoted as r5)
4. Rosenbrock function with 10 inputs (denoted as r10)

Each folder contains json files whose names are appended as:
- \<function\>_\<H\>X for all models used to study the effect of grid-size on optimization effort
- \<function\>_\<H\>X for all models used to study the effect of number of neurons in one hidden layer on optimization effort
- \<function\>_\<H\>X for all models used to study the effect of number of layers on optimization effort

For example, a file named as `Peaks_H1_N2_G6.json` corresponds to a KAN with one hidden layer, two neurons in the hidden layer with six grid points modelling the peaks function.
