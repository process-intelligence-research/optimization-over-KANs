# optimization-over-KANs by [<img src="./docs/logos/Process_Intelligence_Black_Horizontal.png" alt="Process Intelligence Research logo" height="40">](https://www.pi-research.org/)

<p align="center">
  <img src="./docs/logos/pydexpi_logo.png" alt="pyDEXPI logo" width="700">  
</p>

## Overview

This repository contains the [Pyomo](https://github.com/Pyomo/pyomo) files describing the proposed Mixed-Integer Nonlinear Programming formulation in the paper **`Deterministic Global Optimization over trained Kolmogorov Arnold Networks`** (TODO: Add the link to preprint once submitted). <br>
In addition, the repository also contain Python scripts to train multi-layer perceptrons (MLP) using Tensorflow and then optimizing over the trained MLPs using [OMLT](https://github.com/cog-imperial/OMLT). <br>
Effectively, this repository contains all the files needed to reproduce the results in the paper:<br>
(TODO: Add a BibTeX reference to the pre-print) <br>
If you use the formulation from this paper, please consider citing it as described above. <br>

### Features:
- **XXX** as [Pyomo](https://github.com/Pyomo/pyomo) models.

### Citation
Please reference this software package as:
```
@InProceedings{pyDEXPI,
  author    = { and Schweidtmann, Artur M},
  booktitle = {},
  title     = {},
  year      = {2025},
  address   = {},
  month     = {},
}
```

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


## Contributors

| | | |
| --- | --- | --- |
| <img src="https://github.com/user-attachments/assets/d2585cf2-58b5-46db-af7d-672bf3d7501e" width="150" height="100" /> | [Tanuj Karia](https://www.pi-research.org/author/tanuj-karia/) | <a href="https://www.linkedin.com/in/tanujkaria/" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" >  </a> <a href="https://scholar.google.com/citations?user=xNjNE2cAAAAJ&hl=en" rel="nofollow noreferrer"> <img src="docs/logos/google-scholar-square.svg" width="14">  </a> |
| <img src="docs/photos/Artur.jpg" width="50"> | [Artur M. Schweidtmann](https://www.pi-research.org/author/artur-schweidtmann/) | <a href="https://www.linkedin.com/in/schweidtmann/" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" >  </a> <a href="https://scholar.google.com/citations?user=g-GwouoAAAAJ&hl=en" rel="nofollow noreferrer"> <img src="docs/logos/google-scholar-square.svg" width="14">  </a> |

## Copyright and license

This repository is published under MIT license (see [license file](LICENSE))

Copyright (C) 2025 Artur Schweidtmann Delft University of Technology. 

## Contact

📧 [Contact](mailto:a.schweidtmann@tudelft.nl)

🌐 [PI research](https://pi-research.org)

<p align="left">
<a href="https://twitter.com/ASchweidtmann" target="blank"><img align="center" src="https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white" alt="fernandezbap" /></a>
</p>
