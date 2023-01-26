from sympy import *

def Taylor_polynomial_sympy(function_expression, variable_list, evaluation_point, degree):
    """
    Returns a Sympy expression of the Taylor series up to a given degree, of a given multivariate expression, approximated as a multivariate polynomial evaluated at the evaluation_point
    """
    from sympy import factorial, Matrix, prod
    import itertools

    n_var = len(variable_list)
    point_coordinates = [(i, j) for i, j in (zip(variable_list, evaluation_point))]  # list of tuples with variables and their evaluation_point coordinates, to later perform substitution

    deriv_orders = list(itertools.product(range(degree + 1), repeat=n_var))  # list with exponentials of the partial derivatives
    deriv_orders = [deriv_orders[i] for i in range(len(deriv_orders)) if sum(deriv_orders[i]) <= degree]  # Discarding some higher-order terms
    n_terms = len(deriv_orders)
    deriv_orders_as_input = [list(sum(list(zip(variable_list, deriv_orders[i])), ())) for i in range(n_terms)]  # Individual degree of each partial derivative, of each term

    polynomial = 0
    for i in range(n_terms):
        partial_derivatives_at_point = function_expression.diff(*deriv_orders_as_input[i]).subs(point_coordinates)  # e.g. df/(dx*dy**2)
        denominator = prod([factorial(j) for j in deriv_orders[i]])  # e.g. (1! * 2!)
        distances_powered = prod([(Matrix(variable_list) - Matrix(evaluation_point))[j] ** deriv_orders[i][j] for j in range(n_var)])  # e.g. (x-x0)*(y-y0)**2
        polynomial += partial_derivatives_at_point / denominator * distances_powered
    return polynomial

def all_coeffs(expr,variables):
    # Returns a dictionary of all monomials of multivariate polynomials. (Extends the sympy function all_coeffs() to multivariate polynomials)
    x = IndexedBase('x')
    expr = expr.expand()
    free = list(variables)
    pows = [p.as_base_exp() for p in expr.atoms(Pow,Symbol)]
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
    return dict([(m.xreplace(reps), e.coeff(m).xreplace(xzero) if m!=1 else e.xreplace(xzero)) for m in monoms(*[P[f] for f in free])])

def monoms(*o):
    # Used for all_coeffs
    x = IndexedBase('x')
    f = []
    for i,o in enumerate(o):
        f.append(Poly([1]*(o+1),x[i]).as_expr())
    return Mul(*f).expand().args
    