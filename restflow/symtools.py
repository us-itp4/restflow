import itertools
import sympy

def taylor(function_expression, variable_list, evaluation_point, degree):
    """
    Returns a sympy expression of the Taylor series up to a given degree, of
    a given multivariate expression, approximated as a multivariate
    polynomial evaluated at the evaluation_point
    """
    n_var = len(variable_list)
    point_coordinates = [(i, j) for i, j in (zip(variable_list, evaluation_point))]  # list of tuples with variables and their evaluation_point coordinates, to later perform substitution

    deriv_orders = list(itertools.product(range(degree + 1), repeat=n_var))  # list with exponentials of the partial derivatives
    deriv_orders = [deriv_orders[i] for i in range(len(deriv_orders)) if sum(deriv_orders[i]) <= degree]  # Discarding some higher-order terms
    n_terms = len(deriv_orders)
    deriv_orders_as_input = [list(sum(list(zip(variable_list, deriv_orders[i])), ())) for i in range(n_terms)]  # Individual degree of each partial derivative, of each term

    polynomial = 0
    for i in range(n_terms):
        partial_derivatives_at_point = function_expression.diff(*deriv_orders_as_input[i]).subs(point_coordinates)  # e.g. df/(dx*dy**2)
        denominator = sympy.prod([sympy.factorial(j) for j in deriv_orders[i]])  # e.g. (1! * 2!)
        distances_powered = sympy.prod([(sympy.Matrix(variable_list) - sympy.Matrix(evaluation_point))[j] ** deriv_orders[i][j] for j in range(n_var)])  # e.g. (x-x0)*(y-y0)**2
        polynomial += partial_derivatives_at_point / denominator * distances_powered
    return polynomial

def all_coeffs(expr,variables):
    """
    Returns a dictionary of all monomials of multivariate polynomials.
    (Extends the sympy function all_coeffs() to multivariate polynomials)
    """
    x = sympy.IndexedBase('x')
    expr = expr.expand()
    free = list(variables)
    pows = [p.as_base_exp() for p in expr.atoms(sympy.Pow,sympy.Symbol)]
    P = {}
    for p,e in pows:
        if p not in free:
            continue
        elif p not in P:
            P[p]=e
        elif e>P[p]:
            P[p] = e
    reps = dict([(f, x[i]) for i,f in enumerate(free)])
    xzero = dict([(v,0) for k,v in reps.items()])
    e = expr.xreplace(reps); reps = {v:k for k,v in reps.items()}
    return dict([(m.xreplace(reps), e.coeff(m).xreplace(xzero) if m!=1 else e.xreplace(xzero)) for m in _monoms(*[P[f] for f in free])])

def _monoms(*o):
    # Used for all_coeffs
    x = sympy.IndexedBase('x')
    f = []
    for i,o in enumerate(o):
        f.append(sympy.Poly([1]*(o+1),x[i]).as_expr())
    return sympy.Mul(*f).expand().args

def func_coeff(polynomial, q, p):
    """
    Returns a dictionary of the monomials of the integrals
    """    
    # prepare
    cs_psi = sympy.symbols('cos_psi')
    polynomial = sympy.Poly(polynomial,(q.sym,p.sym,cs_psi)).as_expr()
    variables = [q.sym,p.sym,cs_psi]
    var_temp = []
    # following for loop removes the dependency of the polynomial on q,p or
    # cosψ if these monomials do not exist e.g. q^2+q has no p monomial
    for var in variables:
        if var not in polynomial.free_symbols:
            var_temp.append(var)
    variables = [element for element in variables if element not in var_temp]
    coeffs = all_coeffs(polynomial,variables)
    return coeffs

def nullify_outlegs(exprs, k, q, p, r):
    """Sets external outgoing legs to zero for array with expr class objects"""
    dot_pk = k.ctx.dots[frozenset((p.sym,k.sym))]
    dot_qp = k.ctx.dots[frozenset((q.sym,p.sym))]    
    dot_qr = k.ctx.dots[frozenset((q.sym,r.sym))]    
    dot_pr = k.ctx.dots[frozenset((p.sym,r.sym))]    
    dot_rk = k.ctx.dots[frozenset((r.sym,k.sym))]    
    for element in exprs:
        element.num = element.num.subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)])
        element.den = element.den.subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)])
    return exprs