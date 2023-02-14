import sympy
from restflow import symtools

# global symbols
K_d, Lambda, delta_l, dim = sympy.symbols('K_d Lambda δl d')

def _integrate_theta(expr,cs,d):
    """Symbolically replace powers of cos by integrated expression."""
    expr = expr.subs(cs**4,3/(d*(d+2)))
    expr = expr.subs(cs**3,0)
    expr = expr.subs(cs**2,1/d)
    expr = expr.subs(cs,0)
    return expr

def _integrate_magnitude(expr,k,d):
    """
    Calculates the magnitude wavevector integral and multiplies by the
    surface area of hypersphere S_d (from angular integral)
    """
    expr = expr.subs(k.sym,Lambda)*Lambda**d*delta_l*K_d
    return sympy.simplify(expr)

class Expression:
    def __init__(self,num,den):
        self.num = num
        self.den = den

    def integrate1(self,n,k,q):
        """Perform symbolic angular integration of k with one fixed q.

        Expands up to order n of external wave vector q. Only treats scalar
        products up to power of 4.

        Arguments:
            n (int): expansion order
            k (Vector): integration wave vector
            q (Vector): external wave vector

        Returns:
            sympy: scalar symbolic expression
        """
        # prepare
        cs = sympy.Symbol('cos_theta')
        dot = k.ctx.dots[frozenset((q.sym,k.sym))]
        num = sympy.sympify(self.num)
        den = sympy.sympify(self.den)
        expr = num/den
        # replace dot product
        expr = expr.subs(dot,q.sym*k.sym*cs)
        # expand in orders of q
        expr = sympy.series(expr,q.sym,x0=0,n=n).removeO()
        expr = sympy.cancel(expr)
        expr = sympy.Poly(expr,q.sym).as_expr()
        # integrate
        expr = _integrate_theta(expr,cs,dim)
        return _integrate_magnitude(expr,k,dim)

    def integrate2(self,n,k,q,p):
        """Perform symbolic angular integration of k with two external wave
        vectors.

        Arguments:
            n (int): expansion order
            k (Vector): integration wave vector
            q (Vector): external incoming
            p (Vector): external outgoing

        Returns:
            sympy: scalar symbolic expression
        """
        O = sympy.O
        # prepare
        cs_psi, si_psi = sympy.symbols('cos_psi sin_psi')
        cs, x = sympy.symbols('cos_theta (sin_theta·cos_phi)')
        dot_qk = k.ctx.dots[frozenset((q.sym,k.sym))]
        dot_pk = k.ctx.dots[frozenset((p.sym,k.sym))]
        dot_qp = k.ctx.dots[frozenset((q.sym,p.sym))]
        num = self.num
        den = self.den
        num, den = num.subs(dot_qk,q.sym*k.sym*cs), den.subs(dot_qk,q.sym*k.sym*cs)
        num, den = num.subs(dot_pk,p.sym*k.sym*(cs_psi*cs+si_psi*x)), den.subs(dot_pk,p.sym*k.sym*(cs_psi*cs+si_psi*x))
        num, den = num.subs(dot_qp,q.sym*p.sym*cs_psi), den.subs(dot_qp,q.sym*p.sym*cs_psi)
        # expand the expression before the angular integration
        # cancel common factors like q**2 before taylor expansion wrt q
        num = sympy.expand(num)
        den = sympy.expand(den)

        num = num + O(q.sym**n)+O(p.sym**n)+sum([O(q.sym**i*p.sym**(n-i)) for i in range(0,n)])
        num = num.removeO()
        den = den + O(q.sym**n)+O(p.sym**n)+sum([O(q.sym**i*p.sym**(n-i)) for i in range(0,n)])
        den = den.removeO()

        expr = num*symtools.taylor(den**(-1), [q.sym,p.sym], [0,0], n)
        # expand express discarding higher order terms
        expr = sympy.expand(expr)+O(q.sym**n) + O(p.sym**n) + sum([O(q.sym**i*p.sym**(n-i)) for i in range(0,n)])
        expr = expr.removeO()
        # keep only the cs_psi dependence
        expr = expr.subs(si_psi**2,1-cs_psi**2)
        # integrate
        expr = expr.subs(x**2,1/dim)
        expr = expr.subs(x,0)
        # treat remaining cos_theta
        expr = _integrate_theta(expr,cs,dim)       
        # Factorizes the expression in powers of q and p
        expr = sympy.Poly(expr,q.sym,p.sym).as_expr()
        return _integrate_magnitude(expr,k,dim)

def integrate(exprs,n,k,q,p=None):
  """
  Calculates the integrals of all the symbolic graph expressions and
  adds them up.

  Arguments:
    exprs (list): of symbolic.Expression
    k (Vector): integration variable
    labels (list): of outgoing wave vectors
    n (int): expansion order
  """
  if p is None:
    arr = [expr.integrate1(n,k,q) for expr in exprs]
    return sympy.simplify(sympy.Poly(sum(arr)/len(exprs),q.sym).as_expr())
  else:
    arr = [expr.integrate2(n,k,q,p) for expr in exprs]
    return sympy.simplify(sympy.Poly(sum(arr)/len(exprs),q.sym,p.sym).as_expr())
