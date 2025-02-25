# optimization-over-KANs by [<img src="./docs/logos/Process_Intelligence_Black_Horizontal.png" alt="Process Intelligence Research logo" height="40">](https://www.pi-research.org/)

<p align="center">
  <img src="./docs/logos/pydexpi_logo.png" alt="pyDEXPI logo" width="700">  
</p>

## Overview

This repository contains the [Pyomo](https://github.com/Pyomo/pyomo) files describing the proposed Mixed-Integer Nonlinear Programming formulation in the paper **`Deterministic Global Optimization over trained Kolmogorov Arnold Networks`** (TODO: Add the link to preprint once submitted). <br>
In addition, the repository also contain Python scripts to train multi-layer perceptrons (MLP) using Tensorflow and then optimizing over the trained MLPs using [OMLT](https://github.com/cog-imperial/OMLT). <br>
Effectively, this repository contains all the files needed to reproduce the results in the paper:<br>
(TODO: Add a BibTeX reference to the pre-print) <br>
If you use the formulation from this paper, please consider citing it as described below. <br>

### Folder structure:
- **src** contains all [Pyomo](https://github.com/Pyomo/pyomo) files required to create a Pyomo model object of a trained KAN.
- **util** contains all scripts required to reproduce the results in the paper relating to data generation, training of KAN or MLP models.
- **data** contains all training and testing datasets used for training the models in addition to the scaler files in JSON format required for optimizing MLPs using [OMLT](https://github.com/cog-imperial/OMLT).
- **models** contains all KAN models in JSON format which are required to instantiate a Pyomo model object and all MLP models in Keras format.



## Contributors

|  | Name | Links |
| --- | --- | --- |
| <img src="https://github.com/user-attachments/assets/65612774-b784-4a37-b5ba-8430d046a723" width="100" height="100" /> | [Tanuj Karia](https://www.pi-research.org/author/tanuj-karia/) | <a href="https://www.linkedin.com/in/tanujkaria/" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" width="20"> </a> <a href="https://scholar.google.com/citations?user=xNjNE2cAAAAJ&hl=en" rel="nofollow noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Google_Scholar_logo.svg/512px-Google_Scholar_logo.svg.png?20200110094142" width="20"> </a> |
| <img src="https://github.com/user-attachments/assets/b8ad6d34-356a-44be-b34a-d36ae3919fd2" width="100" height="100" /> | [Giacomo Lastrucci](https://www.pi-research.org/author/giacomo-lastrucci/) | <a href="https://www.linkedin.com/in/giacomo-lastrucci/" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" width="20"> </a> <a href="https://scholar.google.com/citations?user=P0_vdtQAAAAJ&hl=en" rel="nofollow noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Google_Scholar_logo.svg/512px-Google_Scholar_logo.svg.png?20200110094142" width="14"> </a> |
| <img src="https://github.com/user-attachments/assets/021e7648-2f69-4db4-a50a-ddb4d409ce5e" width="100" height="100"> | [Artur M. Schweidtmann](https://www.pi-research.org/author/artur-schweidtmann/) | <a href="https://www.linkedin.com/in/schweidtmann/" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" width="20"> </a> <a href="https://scholar.google.com/citations?user=g-GwouoAAAAJ&hl=en" rel="nofollow noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Google_Scholar_logo.svg/512px-Google_Scholar_logo.svg.png?20200110094142" width="14"> </a> |

## Reference
If you use this package, please reference this software package as:
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


## Copyright and license

This repository is published under MIT license (see [license file](LICENSE))

Copyright (C) 2025 Artur Schweidtmann Delft University of Technology. 

## Contact

📧 [Contact](mailto:a.schweidtmann@tudelft.nl)

🌐 [PI research](https://pi-research.org)

<p align="left">
<a href="https://twitter.com/ASchweidtmann" target="blank"><img align="center" src="https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white" alt="fernandezbap" /></a>
</p>
