"""Implementation of symbolic vectors with sympy."""

import sympy
from sympy import *
from symtools import *

dots = {}

class VectorAdd:
    """
    Represents the sum of two vectors or another sum.
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __add__(self, rhs):
        if type(rhs) is Vector or VectorAdd:
            return VectorAdd(self, rhs)
        else:
            raise TypeError(rhs)

    def __sub__(self, rhs):
        if type(rhs) is Vector or VectorAdd:
            return VectorAdd(self, -1*rhs)
        else:
            raise TypeError(rhs)

    def __mul__(self, rhs):
        if type(rhs) is Vector or type(rhs) is VectorAdd:
            return self.a*rhs + self.b*rhs
        else:
            return VectorAdd(self.a*rhs, self.b*rhs)

    def __rmul__(self, lhs):
        return VectorAdd(lhs*self.a, lhs*self.b)

    def __neg__(self):
        return VectorAdd(-1*self.a, -1*self.b)

    def __pow__(self, p):
        if p == 2:
            return self*self
        elif p == 4:
            return (self*self)**2
        elif p == 0:
            return 1
        else:
            raise ValueError

class Vector:
    """
    Class to represent a symbolic vector. Product will return the scalar
    product as symbol. For two vectors x and y also needs symbol in
    dictionary dots to represent dot product.
    """
    def __init__(self, sym, factor=1):
        self.sym = sym
        self.factor = factor

    def __add__(self, rhs):
        if type(rhs) is Vector or type(rhs) is VectorAdd:
            return VectorAdd(self, rhs)
        else:
            raise TypeError(rhs)

    def __sub__(self, rhs):
        if type(rhs) is Vector or type(rhs) is VectorAdd:
            return VectorAdd(self, -1*rhs)
        else:
            raise TypeError(rhs)

    def __mul__(self, rhs):
        if type(rhs) is Vector:
            if self.sym == rhs.sym:
                return self.factor*self.sym * rhs.factor*rhs.sym
            else:
                key = frozenset((self.sym,rhs.sym))
                return self.factor*rhs.factor * dots[key]
        elif type(rhs) is VectorAdd:
            return rhs.a*self + rhs.b*self
        else:
            return Vector(self.sym, rhs*self.factor)

    def __rmul__(self, lhs):
        return Vector(self.sym, lhs*self.factor)

    def __neg__(self):
        return Vector(self.sym, -1*self.factor)

    def __pow__(self, p):
        if p == 2:
            return self*self
        elif p == 4:
            return (self*self)**2
        elif p == 0:
            return 1
        else:
            raise ValueError

def _integrate_theta(expr, cs, d):
    # replace powers of cos by integral
    expr = expr.subs(cs**4,3/(d*(d+2)))
    expr = expr.subs(cs**3,0)
    expr = expr.subs(cs**2,1/d)
    expr = expr.subs(cs,0)
    return expr

def integrate2(expr, k, q, d, n):
    """
    Perform symbolic angular integration of k with fixed q. Expands up to
    order n of external wave vector q. Only treats scalar products up to
    power of 4.
    """
    # prepare
    cs = sympy.Symbol('cos_theta')
    dot = dots[frozenset((q.sym,k.sym))]
    num = sympify(expr[0])
    denum = sympify(expr[1])
    expr = num/denum
    expr = expr.subs(dot,q.sym*k.sym*cs) # replace dot product
    expr = sympy.series(expr,q.sym,x0=0,n=n).removeO()    # expand in orders of q
    expr = cancel(expr)
    expr = Poly(expr,q.sym).as_expr()

    # integrate
    return _integrate_theta(expr,cs,d)

def integrate3(expr, k, q, p, d, n):
    """
    Perform symbolic angular integration of k with two external wave
    vectors.
    """
    # prepare
    from IPython.display import display
    from sympy import Mul
    cs_psi, si_psi = sympy.symbols('cos_psi sin_psi')
    cs, x = sympy.symbols('cos_theta (sin_theta·cos_phi)')
    dot_qk = dots[frozenset((q.sym,k.sym))]
    dot_pk = dots[frozenset((p.sym,k.sym))]
    dot_qp = dots[frozenset((q.sym,p.sym))]
    num = expr[0]
    denum = expr[1]
    num, denum = num.subs(dot_qk,q.sym*k.sym*cs), denum.subs(dot_qk,q.sym*k.sym*cs)
    num, denum = num.subs(dot_pk,p.sym*k.sym*(cs_psi*cs+si_psi*x)), denum.subs(dot_pk,p.sym*k.sym*(cs_psi*cs+si_psi*x))
    num, denum = num.subs(dot_qp,q.sym*p.sym*cs_psi), denum.subs(dot_qp,q.sym*p.sym*cs_psi)
    # expand the expression before the angular integration
    # cancel common factors like q**2 before taylor expansion wrt q
    expr = cancel(num/denum)
    num = fraction(expr)[0]
    denum = fraction(expr)[1]
    
    num = num + O(q.sym**n)+O(p.sym**n)+sum([O(q.sym**i*p.sym**(n-i)) for i in range(0,n)])
    num=num.removeO()
    denum = denum + O(q.sym**n)+O(p.sym**n)+sum([O(q.sym**i*p.sym**(n-i)) for i in range(0,n)])
    denum=denum.removeO()

    expr = num*Taylor_polynomial_sympy(denum**(-1), [q.sym,p.sym], [0,0], n)
    # expand express discarding higher order terms
    expr= expand(expr)+O(q.sym**n)+O(p.sym**n)+sum([O(q.sym**i*p.sym**(n-i)) for i in range(0,n)])
    expr=expr.removeO()
    # keep only the cs_psi dependence
    expr = expr.subs(si_psi**2,1-cs_psi**2)
    # integrate
    expr = expr.subs(x**2,1/d)
    expr = expr.subs(x,0)
    # treat remaining cos_theta
    expr = _integrate_theta(expr,cs,d)       
    # Factorizes the expression in powers of q and p
    expr = Poly(expr,q.sym,p.sym).as_expr()
    return expr

def integrate_magnitude(expr, k, d, *args):
    '''
    Calculates the magnitude wavevector integral and multiplies by the surface area of hypersphere S_d (from angular integral)
    '''
    K_d, Lambda, delta_l = sympy.symbols('K_d, Lambda, δl')
    expr = expr.subs(k.sym, Lambda)*Lambda**d*delta_l*K_d
    return simplify(expr)

def func_coef(polynomial, q, p):
    """
    Returns a dictionary of the monomials of the integrals
    """    
    # Prepare
    cs_psi = sympy.symbols('cos_psi')
    polynomial=Poly(polynomial,(q.sym,p.sym,cs_psi)).as_expr()
    variables=[q.sym,p.sym,cs_psi]
    var_temp=[]
    # Following for loop removes the dependency of the polynomial on q,p or cosψ  if these monomials do not exist e.g. q^2+q has no p monomial
    for var in variables:
        if var not in polynomial.free_symbols:
            var_temp.append(var)
    variables=[element for element in variables if element not in var_temp]
    coeffs=all_coeffs(polynomial,variables)
    return coeffs