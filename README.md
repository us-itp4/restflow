# RESTFLOW

`restflow` is a Python package, implemented using Sympy, for calculating symbolically the flow equations based on Wilsonian Renormalization Group Theory made for applications in statistical physics.

- [Overview](#overview)
- [System requirements](#system-requirements)
- [Installation](#installation)
- [Description of scripts](#description-of-scripts)

## Overview

`restflow` employs sympy to calculate the flow equations for a given dynamical system (equilibrium or non-equilibrium). Given the feynman diagrams, it symbolically solves the integrals and extracts their contributions to the vertex functions. This allows it to overcome tedious calculations arising from perturbation theory. It applies Wilsonian renormalization group theory for a shell of momenta to obtain continuous flow equations. For more information, the following review summarizes the underlying theory and the notations used [TODO: Inserts manuscript].

## System requirements

### Hardware requirements

`restflow` can be run on a standard personal computer. It has been tested on the following setup (without GPU):

+ CPU 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz

### Software requirements

This package has been tested on the following systems with 3.10.9:

+ Windows 10:

`restflow` mainly depends on the following Python packages:

* sympy
* matplotlib
* numpy
* scipy
* feynman

## Installation


#### Directly from the source: clone `restflow` from GitHub
```bash
https://github.tik.uni-stuttgart.de/ac141876/restflow.git
```

## Getting started

The package can be used by importing the `restflow` module and its submodules:
```python
import restflow
```

## Description of scripts

The important scripts of the package are located in `/restflow`

* `symtools.py`: Extends functions of `sympy` (e.g. Taylor expansions) for multivariate functions.
* `symvec.py`: Implements symbolically vectors using sympy. It further symbolically solves the Feynman diagrams
* `graph.py`: Maps a graphical Feynman diagram into an integral. It has option to extract the graph into a LaTeX file.
* `integrals.py`: Iterates and solves the `graph.py` for all the possible graphs given a symmetrized Feynman diagram (by permutating the external leg labels). 
* `neural_networks.ipynb`: Applies `restflow` for a Neural Network Model.
* `example.ipynb`: Applies `restflow` for 2 simple cases of the KPZ model.