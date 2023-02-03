import sympy
import matplotlib.pyplot as plt
import feynman
from restflow import symvec

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

  def link_vertex(self,v,angle=0.0):
    """Link this vertex to another vertex.

    Arguments:
      v (Vertex): target vertex
      angle (real): orientation hint for edge in units of 2π
    """
    e = Edge(start=self,end=v,angle=angle)
    self._out.append(e)
    v._in.append(e)

  def add_outgoing(self,angle=0.0):
    e = Edge(start=self,angle=angle)
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
        e.end.render(diagram,ve)

class Graph:
  """Represents a single graph composed of vertices.

  Attributes:
    vertices (list): list of vertices with the root at position 0
  """
  def __init__(self,vertices):
    self.vertices = vertices
    self.root = vertices[0]
    self.root._in.append(Edge(end=self.root)) # add incoming edge

  def label_edges(self,k,p):
    """Label all edges with the corresponding wave vector.

    Arguments:
      k (Vector): integration variable
      p (list): list of n outgoing wave vectors (Vector)
    """
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

  def freq_integral(self,f,D,k,alpha,a):
    """Calculates the frequency integral of the wavector.

    Arguments:
      ...

    Returns:
      The resulted integral
    """
    propagators, majority, minority=[], [], []
    for item in self.vertices:
      # makes list of intermediate propagators (not external momenta)
      if len(item._in)==1 and item._in[0].end!=None and item._in[0].start!=None:
        propagators.append(item._in[0].label)
    prod_k_prop = [item*k for item in propagators]
    # extract the sign of the k wavector
    # signs=[sympy.Poly(item,k.sym).coeffs()[0] for item in prod_k_prop]
    _ksep = sympy.symbols('k')
    signs = [item.vertices(_ksep**2) for item in prod_k_prop]
    majority = [i for i, x in enumerate(signs) if x==max(set(signs), key = signs.count)]
    minority = [i for i, x in enumerate(signs) if x==min(set(signs), key = signs.count)]
    majority.extend([0,0,0,0]), minority.extend([0,0,0,0]), propagators.extend([0,0,0,0])
    # extracting the alpha exponent of our model

    #dictionary where key is pair (number of propagators, sum of signs) 
    # e.g. (2,0) = Qpm, (3,1) = Qppm
    S0 = (D,f(k, alpha, a))
    Qpm = ((2*f(k, alpha, a)+f(propagators[0], alpha, a)+f(propagators[1], alpha, a)),(f(propagators[0], alpha, a)+f(propagators[1], alpha, a)))
    Qppm = ((2*f(k, alpha, a)*(f(k, alpha, a)+f(propagators[majority[0]], alpha, a)+f(propagators[majority[1]], alpha, a)+f(propagators[minority[0]], alpha, a))+f(propagators[minority[0]], alpha, a)**2+f(propagators[majority[0]], alpha, a)*f(propagators[majority[1]], alpha, a)+f(propagators[majority[0]], alpha, a)*f(propagators[minority[0]], alpha, a)+f(propagators[majority[1]], alpha, a)*f(propagators[minority[0]], alpha, a)),((f(propagators[majority[0]], alpha, a)+f(propagators[minority[0]], alpha, a))*(f(propagators[majority[1]], alpha, a)+f(propagators[minority[0]], alpha, a))))
    dict_freq = {
    (0,0): S0, 
    # (1,1): S0*k**alpha*(propagators[0])**2/(f(propagators[0], alpha, a)+f(k, alpha, a)), 
    (1,1): (S0[0]*k**alpha,S0[1]*((f(propagators[0], alpha, a)+f(k, alpha, a)))), 
    # (2,2): S0*k**alpha*(propagators[0])**2*(propagators[1])**2/((f(k, alpha, a)+f(propagators[0], alpha, a))*(f(k, alpha, a)+f(propagators[1], alpha, a))),
    (2,2): (S0[0]*k**alpha,S0[1]*((f(k, alpha, a)+f(propagators[0], alpha, a))*(f(k, alpha, a)+f(propagators[1], alpha, a)))),
    # (2,0): S0*k**alpha*(propagators[0])**2*(propagators[1])**2/((f(k, alpha, a)+f(propagators[0], alpha, a))*(f(k, alpha, a)+f(propagators[1], alpha, a)))
    # *(2*f(k, alpha, a)+f(propagators[0], alpha, a)+f(propagators[1], alpha, a))/(f(propagators[0], alpha, a)+f(propagators[1], alpha, a)), 
    (2,0): (S0[0]*k**alpha*Qpm[0],S0[1]*((f(k, alpha, a)+f(propagators[0], alpha, a))*(f(k, alpha, a)+f(propagators[1], alpha, a)))*Qpm[1]), 
    # (3,3): S0*k**alpha*(propagators[0])**2*(propagators[1])**2*(propagators[2])**2/((f(k, alpha, a)+f(propagators[0]))*(f(k, alpha, a)+f(propagators[1], alpha, a))*(f(k, alpha, a)+f(propagators[2], alpha, a))), 
    (3,3): (S0[0]*k**alpha,S0[1]*((f(k, alpha, a)+f(propagators[0], alpha, a))*(f(k, alpha, a)+f(propagators[1], alpha, a))*(f(k, alpha, a)+f(propagators[2], alpha, a)))), 
    # (3,1): S0*k**alpha*(propagators[0])**2*(propagators[1])**2*(propagators[2])**2/((f(k, alpha, a)+f(propagators[0], alpha, a))*(f(k, alpha, a)+f(propagators[1], alpha, a))*(f(k, alpha, a)+f(propagators[2], alpha, a)))
    # *(2*f(k, alpha, a)*(f(k, alpha, a)+f(propagators[majority[0]], alpha, a)+f(propagators[majority[1]], alpha, a)+f(propagators[minority[0]], alpha, a))+f(propagators[minority[0]], alpha, a)**2+f(propagators[majority[0]], alpha, a)*f(propagators[majority[1]], alpha, a)+f(propagators[majority[0]], alpha, a)*f(propagators[minority[0]], alpha, a)+f(propagators[majority[1]], alpha, a)*f(propagators[minority[0]], alpha, a))/((f(propagators[majority[0]], alpha, a)+f(propagators[minority[0]], alpha, a))*(f(propagators[majority[1]], alpha, a)+f(propagators[minority[0]], alpha, a)))
    (3,1): (S0[0]*k**alpha*Qppm[0],S0[1]*((f(k, alpha, a)+f(propagators[0], alpha, a))*(f(k, alpha, a)+f(propagators[1], alpha, a))*(f(k, alpha, a)+f(propagators[2], alpha, a)))*Qppm[1])
         }
    return dict_freq[(int(len(signs)),int(abs(sum(signs))))]
 
  def multiplicity(self):
    '''
    Calculates the multiplicity of the symmetrized graph using the formula: N_perm = E!/(B!*W!*(E-B-W)!) where E, B, W are the number of empty, black and white dots for each vertex
    Output: an integer number
    '''
    #TODO Maybe use filters (higher order function)
    import math 
    symmetry=1
    for item in self.vertices:
      num_E=0
      num_B=0
      num_C=0
      for e in item._out:
        if e.end == None:
          num_E+=1
        else:
          if len(e.end._out) == 0:
            num_C+=1
          else:
            num_B+=1
      symmetry*=math.factorial(len(item._out))/(math.factorial(num_E)*math.factorial(num_B)*math.factorial(num_C))
    return int(symmetry)

  def integral(self, f, D, k, alpha, a, v2, v3, *args):
    '''
    Expresses the graph into integral form including its multiplicity.
    Input: Graph (with vertices labelled)
    Output: Integral (sympy expression)
    '''
    I = list(self.freq_integral(f, D, k, alpha, a))
    for v in self.vertices:
      if v.degree == 2:
        I[0] *= v2(v._out[0].label,v._out[1].label,v._in[0].label)[0]
        I[1] *= v2(v._out[0].label,v._out[1].label,v._in[0].label)[1]
      elif v.degree == 3:
        I[0] *= v3(v._out[0].label,v._out[1].label,v._out[2].label, v._in[0].label)[0]
        I[1] *= v3(v._out[0].label,v._out[1].label,v._out[2].label, v._in[0].label)[1]
    I[0] *= self.multiplicity()
    return tuple(I)

  def plot_graph(self):
    '''
    Plots the graph using the feynman package. It is used to verify that the
    input graph of the user is the desired one (cross-check)
    '''
    diagram = feynman.Diagram()
    self.root._render(diagram,diagram.vertex(xy=(.25,.5)))
    diagram.plot()
    plt.show()

  def latex_graph(self,name):
    '''
    Transforms the graph into a TeX file.
    '''
    with open('%s.tex' %name, 'w') as file:
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
