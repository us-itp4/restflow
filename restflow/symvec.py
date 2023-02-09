"""Implementation of symbolic vectors with sympy."""

class Context:
    def __init__(self):
        self.dots = {}

    def add_dot_product(self, s1, s2, s_dot):
        self.dots[frozenset((s1,s2))] = s_dot

    def vector(self,sym):
        return Vector(self,sym)

class VectorAdd:
    """
    Represents the sum of two vectors or another sum.
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @property
    def val(self):
        return self.a.val + self.b.val

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
    context to represent dot product.
    """
    def __init__(self, ctx, sym, factor=1):
        self.ctx = ctx
        self.sym = sym
        self.factor = factor

    @property
    def val(self):
        return self.factor*self.sym

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
            assert(self.ctx == rhs.ctx)
            if self.sym == rhs.sym:
                return self.factor*self.sym * rhs.factor*rhs.sym
            else:
                key = frozenset((self.sym,rhs.sym))
                return self.factor*rhs.factor * self.ctx.dots[key]
        elif type(rhs) is VectorAdd:
            return rhs.a*self + rhs.b*self
        else:
            return Vector(self.ctx, self.sym, rhs*self.factor)

    def __rmul__(self, lhs):
        return Vector(self.ctx, self.sym, lhs*self.factor)

    def __neg__(self):
        return Vector(self.ctx, self.sym, -1*self.factor)

    def __pow__(self, p):
        if p == 2:
            return self*self
        elif p == 4:
            return (self*self)**2
        elif p == 0:
            return 1
        else:
            raise ValueError
