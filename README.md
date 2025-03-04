# Deterministic Global Optimization over Trained Kolmogorov-Arnold Networks (KANs) by [<img src="https://github.com/user-attachments/assets/fa29236a-77fa-44b5-aed6-2c5fe8ce0a0d" height="40">](https://www.pi-research.org/)

This repository contains the [Pyomo](https://github.com/Pyomo/pyomo) files describing the proposed Mixed-Integer Nonlinear Programming (MINLP) formulation in the paper **`Deterministic Global Optimization over trained Kolmogorov Arnold Networks`** (TODO: Add the link to preprint once submitted). 

Additionally, this repository includes:
- Python scripts to train **Multilayer Perceptrons (MLP)** using TensorFlow.
- Optimization routines for trained MLPs using [OMLT](https://github.com/cog-imperial/OMLT).
- Code and data necessary to reproduce all results presented in the paper.
  
---

## Folder structure:
- **src** contains all [Pyomo](https://github.com/Pyomo/pyomo) files required to create a Pyomo model object of a trained KAN.
- **util** contains all scripts required to reproduce the results in the paper relating to data generation, training of KAN or MLP models.
- **data** contains all training and testing datasets used for training the models in addition to the scaler files in JSON format required for optimizing MLPs using [OMLT](https://github.com/cog-imperial/OMLT).
- **models** contains all KAN models in JSON format which are required to instantiate a Pyomo model object and all MLP models in Keras format.

---

## Training and Optimization

### Optimizing Over a Trained KAN
- To optimize over a trained KAN, run:
```sh
python -m opt_kan models/kan/peaks/Peaks_H1_N2_G3.json KAN_formulation_options.json scip
```
#### Arguments:
All the arguments shown in the above example should be passed with the appropriate values.
- `models/kan/peaks/Peaks_H1_N2_G3.json`: Path to the trained KAN model.
- `KAN_formulation_options.json`: Specifies optimization formulation. Refer to the paper for additional details.
- `scip`: Optimization solver.

**Important:** Modify `create_kan.py` (in `src/`) to adjust input variable bounds based on the case study.

### Optimizing Over a Trained MLP
To optimize over a trained MLP, run:
```sh
python -m opt_mlp --keras_model models/mlp/peaks/peaks_mlp_relu_1_16.keras --scaler_file data/peaks_scaler.json --formulation bigm --solver scip --num_inputs 2 --input_lb -3 --input_ub 3 --time_limit 7200
```
#### Key Arguments:
- `--keras_model`: Path to the trained MLP model.
- `--scaler_file`: Path to the JSON file for data scaling.
- `--formulation`: MLP optimization method (e.g., `bigm`).
- `--solver`: Optimization solver.
- `--input_lb`, `--input_ub`: Lower and upper bounds of inputs.
- `--time_limit`: Maximum solver time (seconds).

## Reference
If you use the formulation from this paper, please consider citing it as described below. <br>
```
@article{KANOptimization2025,
  author    = {Tanuj Karia and Giacomo Lastrucci and Artur M. Schweidtmann},
  title     = {Deterministic Global Optimization over Trained Kolmogorov-Arnold Networks},
  journal   = {XX},
  year      = {2025},
  volume    = {XX},
  pages     = {XXX--XXX},
  doi       = {XX.XXXX/XXXXXX}
}
```

## Contributors

|  | Name | Links |
| --- | --- | --- |
| <img src="https://github.com/user-attachments/assets/65612774-b784-4a37-b5ba-8430d046a723" width="100" height="100" /> | [Tanuj Karia](https://www.pi-research.org/author/tanuj-karia/) | <a href="https://www.linkedin.com/in/tanujkaria/" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" width="20"> </a> <a href="https://scholar.google.com/citations?user=xNjNE2cAAAAJ&hl=en" rel="nofollow noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Google_Scholar_logo.svg/512px-Google_Scholar_logo.svg.png?20200110094142" width="20"> </a> |
| <img src="https://github.com/user-attachments/assets/b8ad6d34-356a-44be-b34a-d36ae3919fd2" width="100" height="100" /> | [Giacomo Lastrucci](https://www.pi-research.org/author/giacomo-lastrucci/) | <a href="https://www.linkedin.com/in/giacomo-lastrucci/" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" width="20"> </a> <a href="https://scholar.google.com/citations?user=P0_vdtQAAAAJ&hl=en" rel="nofollow noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Google_Scholar_logo.svg/512px-Google_Scholar_logo.svg.png?20200110094142" width="20"> </a> |
| <img src="https://github.com/user-attachments/assets/021e7648-2f69-4db4-a50a-ddb4d409ce5e" width="100" height="100"> | [Artur M. Schweidtmann](https://www.pi-research.org/author/artur-schweidtmann/) | <a href="https://www.linkedin.com/in/schweidtmann/" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" width="20"> </a> <a href="https://scholar.google.com/citations?user=g-GwouoAAAAJ&hl=en" rel="nofollow noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Google_Scholar_logo.svg/512px-Google_Scholar_logo.svg.png?20200110094142" width="20"> </a> |

## Copyright and license

This repository is published under MIT license (see [license file](LICENSE))

Copyright (C) 2025 Artur Schweidtmann Delft University of Technology. 

## Contact

📧 [Contact](mailto:a.schweidtmann@tudelft.nl)

🌐 [PI research](https://pi-research.org)

<p align="left">
<a href="https://twitter.com/ASchweidtmann" target="blank"><img align="center" src="https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white" alt="fernandezbap" /></a>
</p>
