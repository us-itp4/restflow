import sympy
from restflow import graph
from restflow import symvec

def symmetrize(v, in_legs, out_legs, f, D, k, alpha, a, v2, v3,*args):
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
    g = graph.Graph(v_array[i])
    g.label_edges(in_legs,perm[i],k)
    I_array.append((g.integral(f, D, k, alpha, a, v2, v3)[0], g.integral(f, D, k, alpha, a, v2, v3)[1]*int(len(perm))))
  return I_array

def solve(I_array, k, legs_label, d, n):
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
