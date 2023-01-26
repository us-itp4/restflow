# RESTFLOW

`restflow` is a Python package, implemented using Sympy, for calculating symbolically the flow equations based on Wilsonian Renormalization Group Theory made for applications in statistical physics.

- [Overview](#overview)
- [System requirements](#system-requirements)
- [Installation](#installation)
- [Description of scripts](#description-of-scripts)

## Overview

`restflow` employs sympy to calculate the flow equations for a given dynamical system (equilibrium or non-equilibrium). It automatically solves given feynmann diagrams symbolically and their contributions to the vertex functions. This allows it to overcome tedious calculations with taylor expansions. It applies Wilsonian renormalization group theory for a shell of momenta to obtain continuous flow equations. For more information, the following review summarizes the underlying theory and the notations used [TODO: Inserts manuscript].

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

#### From `pip`:
```bash
pip install restflow
```

#### Or directly from the source: clone `RSMI-NE` from GitHub
```bash
git clone https://github.tik.uni-stuttgart.de/ac141876/restflow
cd RSMI-NE
```
and install the `restflow` package via `pip` in editable mode
```bash
pip install -e .
```
or create a virtual environment and install there:
```bash
./install.sh
```

## Getting started

The package can be used by importing the `rsmine` module and its submodules:
```python
import restflow
```

## Description of scripts

The important scripts of the package are located in `/restflow`

* `symtools.py`: Extends functions of `sympy` (e.g. Taylor expansions) for multivariate functions.
* `symvec.py`: Implements symbolically vectors using sympy. It further symbolically calculates the Feynman diagrams
* `graph.py`: Maps a graphical Feynman diagram into an integral
* `active_model_bplus.ipynb`: Applies `restflow` for Active Model B+.
* `floweqt.ipynb`: For given flow equations, it calculates numerically and graphically the system of equations.