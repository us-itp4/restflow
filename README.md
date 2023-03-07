# RESTFLOW

- [Overview](#overview)
- [Requirements](#requirements)
- [How to use](#how-to-use)

## Overview

`restflow` is a Python package based on [sympy](https://www.sympy.org/) for calculating _symbolically_ the perturbative flow equations of dynamic renormalization. It is a companion package to the manuscript

- Perturbative dynamic renormalization of scalar field theories in statistical physics

When using the package in your academic work, you need to cite this manuscript. For details on the physics and math, consult the manuscript.

At the moment, `restflow` handles 1-loop graphs with complex 2-vertices and simple 3-vertices. A complex vertex function depends on the outgoing wave vectors while simple vertex functions do not contribute wave vectors to the integrand. This is work in progress, be aware that there is almost no checks and error handling.

## Requirements

- sympy
- feynman
- itertools
- math
- IPython.display
- matplotlib.pyplot

## How to use

The best way to learn is probably to look at the examples. To start, import the following
```
import sympy
import restflow
```

### Symbolic vectors

You need to create symbolic wave vectors that label the graph edges. To this end, create a `Context` object and tell it about the dot products that will occur.
```
_k,_q,dot_kq = sympy.symbols('k q (k·q)')   # sympy symbols for vectors
ctx = symvec.Context()
ctx.add_dot_product(_q,_k,dot_kq)
k = ctx.vector(_k)                          # wrap into a symbolic Vector
q = ctx.vector(_q)
```

### The model

The next step is to implement the actual model in a class. This class must implement the following attributes and methods (but only the vertex functions that are needed)
```
class TheModel:
    def __init__(self):
        self.alpha = 0                  # noise exponent
        self.D = sympy.symbols('D')     # noise strength
        # further model parameters...
    
    def f(self,k):
        # from the propagator

    def v2(self,k1,k2,q):
        # vertex function with one incoming and two outgoing lines
        # k1,k2 are the incoming wave vectors and q is their sum

    def v3(self,k1,k2,3,q):
        # vertex function with one incoming and three outgoing lines
```

### Graphs

You now need to create a directed graph. To this end, you create vertices and link them together like this
```
v = [restflow.Vertex() for i in range(3)]
v[0].link_vertex(v[1])                  # add line from v[0] to v[1]
...
v[2].add_outgoing()
```
An outgoing line is added with `add_outgoing()`. Both `link_vertex` and `add_outgoing` except an argument `angle` that specifies the angle with the x axis of the line, which is needed for calling `g.plot_graph()`.

You can now create the graph `g=restflow.Graph(v)`. You need to label the lines, which is achieved by defining first an array `labels = [k, p, q-p]`. The first element of this array is the sink wave vector.  The rest of the arguments is the outgoing wave vectors (in our example composed of `p` and `q`, c.f [here](#symbolic-vectors)), which must agree with the number of outgoing legs. You then call the `g.label_edges(labels)` to label the graph. You should now `g.plot_graph()` to check for mistakes.

### Integrals

If you are satisfied with the graph, you can convert the graph into a symbolic expression calling
```
m = TheModel()
expr = g.convert(m)
```
to which you pass your model definition. Alternatively, you can create a list of symbolic expressions for each permutation of the outgoing legs,
```
exprs = g.convert_perm(m,labels)
```
The final step is to integrate these expression(s)
```
res = restflow.integrate([expr],3,labels)
```
The arguments here are the list of symbolic expressions for the graph (will be summed), the expansion order and the label array.

### Examples

* `kpz.ipynb`: Applies `restflow` to the famous KPZ model.
* `neural_network.ipynb`: Applies `restflow` to a neural network model.
