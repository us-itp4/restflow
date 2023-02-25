import math 
import itertools
import sympy
import matplotlib.pyplot as plt
import feynman
from restflow import symbolic

class Edge:
  """Directed edge with label.

  Attributes:
    start (Vertex): start vertex
    end (Vertex): end vertex
    label (Vector or VectorAdd): wave vector to label edge
    angle (real): orientation hint for edge in units of 2π
  """
  def __init__(self,start=None,end=None,label=None,angle=0.0):
    self.start = start
    self.end = end
    self.label = label
    self.angle = angle

  def _render_label(self):
    return str(self.label.val)

class Vertex:
  """A vertex.

  Attributes:
    _in (list): incoming edges (1 or 2)
    _out (list): outgoing edges (0, 2, or 3)
  """
  def __init__(self):
    self._in = []
    self._out = []

  @property
  def degree(self):
    return len(self._out)

  def link_vertex(self,v,angle=None):
    """Link this vertex to another vertex.

    Arguments:
      v (Vertex): target vertex
      angle (real): orientation hint for edge in units of 2π
    """
    if angle == None:
      e = Edge(start=self, end=v)
    else:
      e = Edge(start=self, end=v, angle=angle)

    self._out.append(e)
    v._in.append(e)

  def add_outgoing(self,angle=None):
    if angle == None: 
      e = Edge(start=self)
    else:
      e = Edge(start=self, angle=angle)

    self._out.append(e)

  def _assign_label(self):
    """
    Recusively assigns the sum of outgoing wave vectors to the incoming leg.
    Call for root vertex.
    """
    sum = None
    for e in self._out:
      if e.label is None:
        e.end._assign_label()
      if sum is None:
        sum = e.label
      else:
        sum = sum + e.label
    self._in[0].label = sum

  def _render(self,diagram,v):
    """
    Recursively add vertices to diagram.
    """
    dx = 0.2
    if self._in[0].start is None:
      v0 = diagram.vertex(v._xy,dxy=(-dx,0))
      l = diagram.line(vstart=v0,vend=v)
      l.text(self._in[0]._render_label(),horizontalalignment='center')
    for e in self._out:
      if e.end and e.end.degree == 0: # is correlation
        if hasattr(e.end,'_g'):
          ve = e.end._g
        else:
          ve = diagram.vertex(v._xy,dxy=(.5*dx,dx),marker='o',markerfacecolor='white',markeredgewidth=2)
          e.end._g = ve
      else:
        ve = diagram.vertex(v._xy,angle=e.angle,radius=dx)
      l = diagram.line(vstart=v,vend=ve)
      l.text(e._render_label(),horizontalalignment='center')
      if e.end:
        e.end._render(diagram,ve)

class Graph:
  """Represents a single graph composed of vertices.

  Attributes:
    k (Vector): internal wave vector
    vertices (list): list of vertices with the root at position 0
  """
  def __init__(self,vertices):
    self.k = None
    self.vertices = vertices
    self.root = vertices[0]
    self.root._in.append(Edge(end=self.root)) # add incoming edge

  def _reset_labels(self):
    """Deletes all the labels of the edges"""
    for vertex in self.vertices:
      for e_in in vertex._in:
        e_in.label = None
      for e_out in vertex._out:
        e_out.label = None

  def label_edges(self,labels):
    """Label all edges with the corresponding wave vector.

    Requires a single-loop graph with a single correlation function.

    Arguments:
      k (Vector or VectorAdd): internal wave vector
      p (list): list of n outgoing wave vectors (Vector or VectorAdd)
    """
    self._reset_labels()  #clears the previous labels to replace them
    k = labels[0]
    self.k = labels[0]
    p = [item for item in labels[1:]]
    p = p.copy()
    # label external legs
    for v in self.vertices:
      if v.degree == 0:     # is correlation function
        v._in[0].label = k
        v._in[1].label = -k
      else:
        for e in v._out:
          if e.end == None: # is leaf
            e.label = p.pop(0)
    # now label all internal edges obeying "momentum conservation"
    self.root._assign_label()

  def _ext_vec(self):
    """Calculates array of square of labels of external legs"""
    ext_array=[]
    for v in self.vertices:
     if v.degree > 0:
      for leaf in v._out:
        if leaf.end == None:
          ext_array.append(leaf.label)
      if v._in[0].start == None:
        ext_array.append(v._in[0].label)
    return [element**2 for element in ext_array]

  def _calculate_freq_integral(self,k,f):
    """Determines the result of the frequency integration.

    Arguments:
      k (Vector): internal wave vector
      f (func): propagator function
      p (list): list of n outgoing wave vectors (Vector)
    Returns:
      tuple: (numerator,denominator)
    """
    k_edge, majority, minority = [], [], []
    for v in self.vertices:
      # makes list of intermediate edge wave vectors excluding external legs
      if v.degree > 0 and v._in[0].end != None and v._in[0].start != None:
        k_edge.append(v._in[0].label)

    #load array of external legs
    ext_array=self._ext_vec()
    #multiply by sink vector
    k2_edge = [_k*k for _k in k_edge]

    # extract the sign of the k wavector
    num_prop = len(k_edge)
    signs = [0]*num_prop
    for i in range(len(k2_edge)):
      k2_edge[i] = k2_edge[i].subs([(external,0) for external in ext_array]) # set external legs^2 to 0
      for term in k2_edge[i].as_ordered_terms():  # for the remaining monomials
          coeff, _ = term.as_coeff_Mul()
          expr=term**2  # to avoid vector complications
          if expr.is_Pow and expr.exp == 4: # if monomial is x^4 
              signs[i]=coeff  # extract its coefficient

    majority = [i for i, x in enumerate(signs) if x==max(set(signs), key = signs.count)]
    minority = [i for i, x in enumerate(signs) if x==min(set(signs), key = signs.count)]
    majority.extend([0,0,0,0]), minority.extend([0,0,0,0]), k_edge.extend([0,0,0,0]) # pad lists with zeros

    # Q-function tuples of (numerator,denominator)
    Qpm = (
      2*f(k)+f(k_edge[0])+f(k_edge[1]),
      f(k_edge[0])+f(k_edge[1])
    )
    Qppm = (
      2*f(k)*(f(k)+f(k_edge[majority[0]])+f(k_edge[majority[1]])+f(k_edge[minority[0]]))+f(k_edge[minority[0]])**2+f(k_edge[majority[0]])*f(k_edge[majority[1]])+f(k_edge[majority[0]])*f(k_edge[minority[0]])+f(k_edge[majority[1]])*f(k_edge[minority[0]]),
      (f(k_edge[majority[0]])+f(k_edge[minority[0]]))*(f(k_edge[majority[1]])+f(k_edge[minority[0]]))
    )

    # dictionary of (numerator,denominator) where key is the pair (number of
    # propagators, sum of signs) e.g. (2,0) = Qpm, (3,1) = Qppm
    dict_freq = {
      (0,0): (1,1), 
      (1,1): (1,f(k_edge[0])+f(k)),
      (2,2): (1,(f(k)+f(k_edge[0]))*(f(k)+f(k_edge[1]))),
      (2,0): (Qpm[0],(f(k)+f(k_edge[0]))*(f(k)+f(k_edge[1]))*Qpm[1]),
      (3,3): (1,(f(k)+f(k_edge[0]))*(f(k)+f(k_edge[1]))*(f(k)+f(k_edge[2]))), 
      (3,1): (Qppm[0],((f(k)+f(k_edge[0]))*(f(k)+f(k_edge[1]))*(f(k)+f(k_edge[2])))*Qppm[1])
    }
    return dict_freq[(len(signs),int(abs(sum(signs))))]

  def calculate_multiplicity(self):
    """Calculates the multiplicity of the symmetrized graph.

    Returns:
      int: the multiplicity
    """
    mult = 1
    for v in self.vertices:
      num_E = 0
      num_B = 0
      num_C = 0
      for e in v._out:
        if e.end == None:
          num_E += 1
        else:
          if len(e.end._out) == 0:
            num_C += 1
          else:
            num_B += 1
      mult *= math.factorial(len(v._out))/(math.factorial(num_E)*math.factorial(num_B)*math.factorial(num_C))
    return int(mult)

  def convert(self, model):
    """Converts the graph into a symbolic expression.

    This methods produces a symbolic representation of the graph, which
    needs to be labeled. It requires a model definition and calculates
    nominator and denominator separately.

    Arguments:
      model: object holding the model definition
      p (list): list of n outgoing wave vectors (Vector)

    Returns:
      Expression: the integrand
    """
    
    num,den = self._calculate_freq_integral(self.k,model.f)
    for v in self.vertices:
      if v.degree == 2:
        v2_num,v2_den = model.v2(v._out[0].label,v._out[1].label,v._in[0].label)
        num *= v2_num
        den *= v2_den
      elif v.degree == 3:
        v3_num,v3_den = model.v3(v._out[0].label,v._out[1].label,v._out[2].label,v._in[0].label)
        num *= v3_num
        den *= v3_den
    num *= self.calculate_multiplicity()*model.D*self.k**model.alpha
    den *= model.f(self.k)
    return symbolic.Expression(num,den)

  def convert_perm(self,model, labels):
    """Converts the graph into a list of symbolic expression.

    Calculates all permutations of outgoing wave vectors p and determines
    their symbol expressions. Note that this method relabels graph edges.

    Arguments:
      model: object holding the model definition
      k (Vector): internal wave vector
      p (list): list of n outgoing wave vectors (Vector)

    Returns:
      list: of Expression objects
    """
    exprs = []
    k = labels[0]
    self.k = labels[0]
    p = [item for item in labels[1:]]
    p = p.copy()
    for _p in itertools.permutations(p):
      self.label_edges([k]+list(_p))
      exprs.append(self.convert(model))
    return exprs

  def plot_graph(self):
    """Plots the graph using the feynman package.

    It can be used to verify that the input graph of the user is the desired
    one.
    """
    diagram = feynman.Diagram()
    self.root._render(diagram,diagram.vertex(xy=(.25,.5)))
    diagram.plot()
    plt.show()

  def export_latex_graph(self,filename):
    """Transforms the graph into a TeX file."""
    with open(f'{filename}.tex','w') as file:
      file.write(r'\documentclass[11pt,a4paper,border={1pt 1pt 16pt 1pt},varwidth]{standalone}' '\n' r'\usepackage[top=15mm,bottom=12mm,left=30mm,right=30mm,head=12mm,includeheadfoot]{geometry}' '\n' r'\usepackage{graphicx,color,soul}' '\n' r'\usepackage[compat=1.1.0]{tikz-feynman}' '\n' r'\usepackage{XCharter}' '\n' r'\begin{document}' '\n' r'\thispagestyle{empty}' '\n' r'\begin{figure*}[t]' '\n \t' r'\hspace{-0.4cm}\feynmandiagram [small,horizontal=root to v0] {' '\n')
      num_leaf=0     
      file.write('\t \t root -- [fermion] v0, \n') 
      for i in range(len(self.vertices)):
        if len(self.vertices[i]._out) != 0:
          # loop through all neighbors of selected vertex
          for j in range(len(self.vertices[i]._out)):
            # extract id of neighboring vertices which are not ends of leaves
            id_out = [self.vertices.index(e.end) if e.end in self.vertices else None for e in self.vertices[i]._out]
            # if it is end of leaf
            if self.vertices[i]._out[j].end == None:
              file.write('\t \t v{0} -- [fermion] v{1}{2}, \n'.format(i, id_out[j], num_leaf))
              num_leaf+=1
            # if it is a sink
            elif len(self.vertices[i]._out[j].end._in) == 2:
              file.write('\t \t v{0} -- [fermion] v{1} [/tikzfeynman/empty dot], \n'.format(i, id_out[j]))            
            else:
              file.write('\t \t v{0} -- [fermion] v{1}, \n'.format(i, id_out[j]))                    
      file.write('\t }; \n')
      file.write('\end{figure*} \n')
      file.write('\end{document}')
