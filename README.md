# AAIS
Source code for Adaptive Adversarial Augmentation for Molecular Property Prediction with pytorch and torch_geometric.


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
We use the seven binary classification benchmark datasets included in [OGB](https://github.com/snap-stanford/ogb): BBBP, CLINTOX, SIDER, HIV, BACE, TOX21, and TOXCAST.
The primary considerations in training include dataset, GNN type (gnn_type), optimizer (optim_method), train_type, and burn-in period.

The default setting specifies GCN as the GNN type, a burn-in period of 20, and SGD as the optimizer. 
In our repository, we support two GNN types (GCN and GIN), and two optimizers (SGD and Adam).

We have consdiered subsampling ratio $r$ of {0, 0.1, 0.3, 0.5, 0.7, 0.9, 1}. Additionally, $r$ can be adjusted to any desired value between 0 and 1. When $r=0$, it implies training without data augmentation, setting the train_type argument to 'base'. Conversely, when $r=1$, it denotes applying adversarial augmentation to all data, setting the train_type argument to 'aa'.
More detailed arguments are summarized in [argument.py](module/argument.py).

- ### Version 1: Training with the fixed subsampling ratio $r$:
```python
# without augmentation (r = 0) 
python main.py --dataset=bbbp --train_type=base

# r = 0.5 
python main.py --dataset=bbbp --train_type=aais --ratio=0.5

# r = 1
python main.py --dataset=bbbp --train_type=aa
```

If you want to add a virtual node,
```python 
python main.py --virtual=True --dataset=bbbp -train_type=aais --ratio=0.5
```

If you have a problem with DataLoader, 
```python
python main.py --dataset=bbbp --num_workers=0
```

- ### Version 2: Tuning the subsampling ratio $r$ during training:
``` python
python main_tuned.py --dataset=bbbp
```
Considering the potentially significant computational time required, carrying out this procedure on a large server is recommended, especially when handling HIV, TOX21, and TOXCAST datasets.


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
├── main_tuned.py
├── requirements.txt
├── example.ipynb
├── README.md
└── .gitignore
```
- module/argument.py: set of arguments
- module/set_seed.py: specify the seed
- module/sgd_influence.py: calculate the one-step influence function when using SGD optimizer
- module/adam_influence.py: calculate the one-step influence function when using Adam optimizer
- module/model.py: model architectures
- module/train.py: training/evaluation functions
- main.py: script for training when the subsampling ratio is fixed
- main_tuned.py: script for training with subsampling ratio tuning
- requirements.txt: dependencies for this repository
- example.ipynb: tutorial for implementation AAIS
