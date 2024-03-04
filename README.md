# AAIS
Source code for Adaptive Adversarial Augmentation for Molecular Property Prediction with pytorch.


## Requiremets & Installation
The code is written in Python 3 (>= 3.9.0) and supports both GPU with cuda 11.3 and CPU on Windows. MacOSX is only supported on the CPU.

1. Clone the Git repository.
2. Install a compatible version of Python and make an environment.
```
conda create -n aais ptyhon=3.9.0
conda activate aais
```
3. Install the dependencies from the requirements file. 
```
pip install -r requirements.txt
```


## Basic Usage
Training using AAIS with the subsampling ratio $r$ of 0.5:
```
python main.py --args.dataset=bbbp --args.train_type=aais --args.ratio=0.5 --args.optim_method=sgd
```


## Components
```
├── module
│   ├── __init__.py
│   ├── argument.py
│   ├── set_seed.py
│   ├── sgd_influence.py
│   ├── adam_influence.py
│   ├── model.py
│   └── train.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```
- module/argument.py: set of arguments
- module/set_seed.py: specify the seed
- module/sgd_influence.py: calculate the one-step influence function when using SGD optimizer
- module/adam_influence.py: calculate the one-step influence function when using Adam optimizer
- module/model.py: model architecture
- module/train.py: training/inference functions
- main.py: script for taining
- requirements.txt: Dependencies for this repository
