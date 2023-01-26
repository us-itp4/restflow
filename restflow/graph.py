import sympy
from restflow import symvec

class Edge:
  """
  Directed edge with label. External edges have 
  """
  def __init__(self,start=None,end=None,label=None):
    self.start = start
    self.end = end
    self.label = label

class Vertex:
  """
  A vertex.
    _in: incoming edges (1 or 2)
    _out: outgoing edges (0, 2, or 3)
  """
  def __init__(self):
    self._in = []
    self._out = []

  def degree(self):
    return len(self._out)
  
  def link_vertex(self,v):
    e = Edge(start=self,end=v)
    self._out.append(e)
    v._in.append(e)

  def add_outgoing(self):
    e = Edge(start=self)
    self._out.append(e)

class Graph:
  """
  Represents a single graph composed of vertices.
  """
  def __init__(self,v):
    self.coeff = v # list of coefficients

  def _leaves_(self,out_legs):
    out_legs_temp=out_legs
    for item in self.coeff:
      for i in range(len(item._out)):
        if item._out[i].end == None:
          item._out[i].label = out_legs_temp[0]
          del out_legs_temp[0]

  def _sink_(self,k):
    for item in self.coeff:
      if len(item._out)==0:
        item._in[0].label = k
        item._in[1].label = -k

  def _root_(self, in_leg):
    for item in self.coeff:
      if len(item._in)==0:
        item._in.append(Edge(end=item,label=in_leg))

  def label_edges(self,in_leg,out_legs,k):
    self._root_(in_leg)
    self._leaves_(out_legs)
    self._sink_(k)
    # label all internal edges obeying "momentum conservation"
    while any(item._in[i].label == None for item in self.coeff for i in range(len(item._in)) ):
      for item in self.coeff:
        if len([item._in[x].label for x in range(len(item._in)) if item._in[x].label == None]) == 1 and len([item._out[x].label for x in range(len(item._out)) if item._out[x].label == None]) == 0: 
          in_labels = [e.label for e in item._in if e.label is not None]
          out_labels = [e.label for e in item._out]
          for e in item._in:
            if e.label == None:
              sum_out = out_labels[0]
              for var in out_labels[1:]:
                sum_out += var
              if len(in_labels) != 0:
                e.label = sum_out-in_labels[0]
              else:
                e.label = sum_out

  def freq_integral(self,f,D,k,alpha,a):
    '''
    Calculates the frequency integral of the wavector.
    Output: The resulted integral
    '''
    propagators, majority, minority=[], [], []
    for item in self.coeff:
      # makes list of intermediate propagators (not external momenta)
      if len(item._in)==1 and item._in[0].end!=None and item._in[0].start!=None:
        propagators.append(item._in[0].label)
    prod_k_prop=[item*k for item in propagators]
    # extract the sign of the k wavector
    # signs=[sympy.Poly(item,k.sym).coeffs()[0] for item in prod_k_prop]
    _ksep =  sympy.symbols('k')
    signs=[item.coeff(_ksep**2) for item in prod_k_prop]
    majority=[i for i, x in enumerate(signs) if x==max(set(signs), key = signs.count)]
    minority=[i for i, x in enumerate(signs) if x==min(set(signs), key = signs.count)]
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
    for item in self.coeff:
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
    for item in self.coeff:
      if len(item._out)== 2:
        I[0]*= v2(item._out[0].label,item._out[1].label,item._in[0].label)[0]
        I[1]*= v2(item._out[0].label,item._out[1].label,item._in[0].label)[1]
      elif len(item._out)== 3:
        I[0]*= v3(item._out[0].label,item._out[1].label,item._out[2].label, item._in[0].label)[0]
        I[1]*= v3(item._out[0].label,item._out[1].label,item._out[2].label, item._in[0].label)[1]
    I[0]*=self.multiplicity()
    return tuple(I)
    
  def plot_graph(self):
    '''
    Plots the graph using the feynman package. It is used to verify that the input graph of the user is the desired one (cross-check)
    '''
    import matplotlib.pyplot as plt
    from feynman import Diagram
    import numpy as np
    diagram = Diagram()
    v0 = diagram.vertex(xy=(.1,.5),marker='')
    # creates array that stores tag of vertex (sink or normal vertex) and its coordinates
    info_vertex= [['empty', v0.xy] for i in range(len(self.coeff))]
    info_vertex[0] = ['vertex', np.array([v0.x+.2, v0.y])]
    # creates array that stores the outgoing (empty) vertices and the id of the connected vertex
    info_leaf = []
    dx = .2
    gap = .15
    dy_3 = [0, gap, -gap]
    dy_2 = [-gap, gap]

    #loop to assign coordinates of all vertices (apart from sink)
    for i in range(len(self.coeff)):
      # array with id_s of the vertices where edges from a chosen vertex end up (if there are any such vertices)
      id_out = [self.coeff.index(item.end) if item.end in self.coeff else None for item in self.coeff[i]._out]
      for j in range(len(self.coeff[i]._out)):
        dy = dy_2 if len(self.coeff[i]._out)==2 else dy_3
        if self.coeff[i]._out[j].end != None:
          info_vertex[id_out[j]][1] = np.array([info_vertex[i][1][0]+dx, info_vertex[i][1][1]+dy[j]])
        elif self.coeff[i]._out[j].end == None:
          info_leaf.append([i, np.array([info_vertex[i][1][0]+dx, info_vertex[i][1][1]+dy[j]])])
      info_vertex[i][0] = 'sink' if len(self.coeff[i]._out) ==0 else 'vertex'
    
    #loop to displace coordinates of outgoing (empty) vertices if same with other vertices
    for leaf in info_leaf:
      if any([(abs(leaf[1]-vertex[1])<1e-5).all() for vertex in info_vertex]):
        leaf[1][1] = leaf[1][1]+np.sign(info_vertex[leaf[0]][1][1]-v0.y)*2*gap
        leaf[1][0] = info_vertex[leaf[0]][1][0]

    #loop to assign coordinates of the sink
    for i in range(len(self.coeff)):
      if len(self.coeff[i]._in) == 2:
        in1, in2 = self.coeff[i]._in[0].start, self.coeff[i]._in[1].start
        index1, index2 = self.coeff.index(in1), self.coeff.index(in2)
        # sink gets assigned next to the vertex closest to the root
        min_in = min(info_vertex[index1][1][0], info_vertex[index2][1][0])
        min_index = index1 if abs(min_in - info_vertex[index1][1][0])<1e-5 else index2
        dy = dy_2 if len(self.coeff[min_index]._out)==2 else dy_3
        potential_out = [np.array([info_vertex[min_index][1][0]+dx, info_vertex[min_index][1][1] + element]) for element in dy]
        #loop to assign coordinates of the sink to the empty spot of the connected vertex
        for pair in potential_out:
          if not any([(abs(pair-vertex[1])<1e-5).all() for vertex in info_vertex]) and not any([(abs(pair-vertex[1])<1e-5).all() for vertex in info_leaf]):
            info_vertex[i][1] = pair
    # creates the vertex array using the information from info_vertex and info_leaf        
    v = [diagram.vertex(item[1]) if item[0] == 'vertex' else diagram.vertex(item[1], marker='o', markerfacecolor='white', markeredgewidth=2) for item in info_vertex]
    v_leaf = [diagram.vertex(item[1], marker='') for item in info_leaf]
    # plot the lines between the vertices
    times=0
    for i in range(len(self.coeff)):
      id_out = [self.coeff.index(item.end) if item.end in self.coeff else None for item in self.coeff[i]._out]
      # loop through the out vertices of i
      for j in range(len(self.coeff[i]._out)):
        # if not a leaf
        if self.coeff[i]._out[j].end != None:
          # if not a sink
          if len(self.coeff[id_out[j]]._in) != 2:
            middle_branches = diagram.line(v[i], v[id_out[j]])
          # if a sink
          else:
            # if a self-loop: uses elliptical lines
            if self.coeff[id_out[j]]._in[0].start == self.coeff[id_out[j]]._in[1].start and times==0:
              middle_branches = diagram.line(v[i], v[id_out[j]], style='elliptic', ellipse_spread=0.25)
              times+=1
            elif self.coeff[id_out[j]]._in[0].start == self.coeff[id_out[j]]._in[1].start and times==1:
              middle_branches = diagram.line(v[i], v[id_out[j]], style='elliptic', ellipse_spread=-0.25)
              times+=1
            # if not a self-loop
            else:
              middle_branches = diagram.line(v[i], v[id_out[j]])

    out_branches = [diagram.line(v[info_leaf[i][0]], v_leaf[i]) for i in range(len(info_leaf))]
    root_branch = diagram.line(v0, v[0])
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
      for i in range(len(self.coeff)):
        if len(self.coeff[i]._out) != 0:
          # loop through all neighbors of selected vertex
          for j in range(len(self.coeff[i]._out)):
            # extract id of neighboring vertices which are not ends of leaves
            id_out = [self.coeff.index(item.end) if item.end in self.coeff else None for item in self.coeff[i]._out]
            # if it is end of leaf
            if self.coeff[i]._out[j].end == None:
              file.write('\t \t v{0} -- [fermion] v{1}{2}, \n'.format(i, id_out[j], num_leaf))
              num_leaf+=1
            # if it is a sink
            elif len(self.coeff[i]._out[j].end._in) == 2:
              file.write('\t \t v{0} -- [fermion] v{1} [/tikzfeynman/empty dot], \n'.format(i, id_out[j]))            
            else:
              file.write('\t \t v{0} -- [fermion] v{1}, \n'.format(i, id_out[j]))                    
      file.write('\t }; \n')
      file.write('\end{figure*} \n')
      file.write('\end{document}')

def integrals_symmet(v, in_legs, out_legs, f, D, k, alpha, a, v2, v3,*args):
  '''
  Calculates graphs by permutating labels of external legs
  Output: Array with the integrals with calculated frequency integrals.
  The final integrals are divided by number of such permutations.
  '''
  import itertools
  import copy
  perm = list(itertools.permutations(out_legs))
  I_array = []
  v_array = [copy.deepcopy(v) for i in range(len(perm))]
  for i in range(len(perm)):
    perm[i] = list(perm[i])
    g = Graph(v_array[i])
    g.label_edges(in_legs,perm[i],k)
    I_array.append((g.integral(f, D, k, alpha, a, v2, v3)[0], g.integral(f, D, k, alpha, a, v2, v3)[1]*int(len(perm))))
  return I_array

def solve_integrals_symmet(I_array, k, legs_label, d, n):
  '''
  Calculates the angular integrals of all the labelled graphs and adds them up.
  '''
  if len(legs_label)==1:
    solved_I = [symvec.integrate2(element, k, legs_label[0], d, n) for element in I_array]
    solved_I = [symvec.integrate_magnitude(element, k, d) for element in solved_I]
    return sympy.Poly(sum(solved_I), legs_label[0].sym).as_expr()
  elif len(legs_label)==2:
    solved_I = [symvec.integrate3(element, k, legs_label[0], legs_label[1], d, n) for element in I_array]
    solved_I = [symvec.integrate_magnitude(element, k, d) for element in solved_I]
    return sympy.Poly(sum(solved_I), legs_label[0].sym, legs_label[1].sym).as_expr()
