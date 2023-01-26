import sympy
import symvec

# dimension and reduced surface area
d, Kd = sympy.symbols('d K_d')
# symbols for vectors
_q, _k, dot_kq = sympy.symbols('q k (k·q)')
# assign symbol for dot product
symvec.dots[frozenset((_q,_k))] = dot_kq
# create vectors
k = symvec.Vector(_k)
q = symvec.Vector(_q)

x = (k*(k-q))**2
print('(k*(k-q))**2) =',sympy.expand(x))

x = symvec.integrate2(x, k, q, d, Kd)
print(x)

