import unittest
import sys
from IPython.display import Image, display
import sympy
import os
script_path = os.path.abspath(__file__)     # Get the absolute path of the script
script_dir = os.path.dirname(script_path)   # Get the directory of the script
os.chdir(script_dir)    # Change the current working directory to the directory of the script
sys.path.append('..')   # path for local package
import restflow
from restflow import symvec
from restflow import symbolic

#parameters of model
c1, c2, c3, u, D, kappa = sympy.symbols('c1 c2 c3 u D kappa')
# dimension and reduced surface area
K_d, Lambda, delta_l, dim = sympy.symbols('K_d Lambda δl d')
# symbols for vectors
_q, _p, _r, _k, dot_kq, dot_pk, dot_qp, dot_qr, dot_pr, dot_rk = sympy.symbols('q p r k (k·q) (k·p) (q·p) (q·r) (p·r) (r·k)')

ctx = restflow.Context()
ctx.add_dot_product(_q, _k, dot_kq)
ctx.add_dot_product(_p, _k, dot_pk)
ctx.add_dot_product(_q, _p, dot_qp)
ctx.add_dot_product(_q, _r, dot_qr)
ctx.add_dot_product(_p, _r, dot_pr)
ctx.add_dot_product(_r, _k, dot_rk)

# create vectors
k, q, p, r = ctx.vector(_k), ctx.vector(_q), ctx.vector(_p), ctx.vector(_r)
'''------------------------------------------------------------------------------------------------------------'''

class AMBp:
    def __init__(self):
        self.alpha = 2
        self.c1, self.c2, self.c3, self.u, self.D, self.kappa = sympy.symbols('c1 c2 c3 u D kappa')

    def f(self,k):
        return (self.kappa*k**self.alpha)

    def v2(self,k1,k2,q):
        expr = (self.c1*q**2+2*self.c2*k1*k2-self.c3*(k2**2*q*k1+k1**2*q*k2)/(q**2))*1/2
        return sympy.fraction(sympy.cancel((expr)))

    def v3(self,k1,k2,k3,q):
        return (-self.u,1)

class Testclass(unittest.TestCase):

    def test_integrate2(self):
        model = AMBp()
        expr = symbolic.Expression(model.c1*q**2,sympy.sympify(2))
        self.assertEqual(expr.integrate1(5, k, q), model.c1*q**2/2*Lambda**dim*delta_l*K_d)
        expr = symbolic.Expression(dot_kq**2*model.c3,q**2)
        self.assertEqual(expr.integrate1(5, k, q), model.c3*Lambda**(dim+2)*delta_l*K_d/dim)
        expr = symbolic.Expression(dot_kq*model.c2,sympy.sympify(1))
        self.assertEqual(expr.integrate1(5, k, q), 0)
        expr = symbolic.Expression((1+dot_kq)*(2+dot_kq),sympy.sympify(1))
        self.assertEqual(expr.integrate1(5, k, q), K_d*Lambda**dim*delta_l*(Lambda**2*q**2 + 2*dim)/dim)
        expr = symbolic.Expression(sympy.sympify(1),(k+q)**2)
        self.assertEqual(expr.integrate1(5, k, q), K_d*Lambda**(dim - 6)*delta_l*(Lambda**4*dim*(dim + 2) - Lambda**2*q**2*(dim - 4)*(dim + 2) + q**4*(dim*(dim + 2) - 12*dim + 24))/(dim*(dim + 2)))

    def test_integrate3(self):
        cs_psi = sympy.symbols('cos_psi')
        expr = symbolic.Expression((q-p-k)*(q-k),sympy.sympify(1))
        self.assertEqual(expr.integrate2(3, k, q, p), K_d*Lambda**dim*delta_l*(Lambda**2 - cs_psi*p.sym*q.sym + q**2))
        expr = symbolic.Expression(((k)*(q+p))**2,sympy.sympify(1))
        self.assertEqual(expr.integrate2(3, k, q, p), K_d*Lambda**(dim + 2)*delta_l*(2*cs_psi*p.sym*q.sym + p**2 + q**2)/dim)
        expr = symbolic.Expression(sympy.sympify(1),(q+k+p)**2)
        self.assertEqual(expr.integrate2(3, k, q, p), K_d*Lambda**(dim - 4)*delta_l*(Lambda**2*dim - (dim - 4)*(2*cs_psi*p.sym*q.sym + p**2 + q**2))/dim)

    def test_label_edges(self):
        #kpz figure
        v = [restflow.Vertex() for i in range(3)]
        v[0].link_vertex(v[1])
        v[0].link_vertex(v[2])
        v[1].link_vertex(v[2])
        v[1].add_outgoing(0)
        g = restflow.Graph(v)
        labels = [k, q]
        g.label_edges(labels)

        n = [restflow.Vertex() for i in range(3)]
        n[0].link_vertex(n[1])
        n[0]._out[0].label = q-k
        n[0].link_vertex(n[2])
        n[0]._out[1].label = k
        n[1].link_vertex(n[2])
        n[1]._out[0].label = -k
        n[1].add_outgoing()
        n[1]._out[1].label = q
        h = restflow.Graph(n)
        h.vertices[0]._in[0].label=q

        for i in range(len(g.vertices)):
            for j in range(len(g.vertices[i]._in)):
                self.assertEqual((g.vertices[i]._in[j].label)**2, (h.vertices[i]._in[j].label)**2)
        
        v = [restflow.Vertex() for i in range(5)]
        v[0].link_vertex(v[1])
        v[0].link_vertex(v[2])
        v[2].link_vertex(v[3])
        v[3].link_vertex(v[4])
        v[4].link_vertex(v[1])
        v[2].add_outgoing()
        v[3].add_outgoing()
        v[4].add_outgoing()
        g = restflow.Graph(v)
        labels = [k, p,r,q-p-r]
        g.label_edges(labels)

        n = [restflow.Vertex() for i in range(5)]
        n[0].link_vertex(n[1])
        n[0]._out[0].label = k
        n[0].link_vertex(n[2])
        n[0]._out[1].label = q-k
        n[2].link_vertex(n[3])
        n[2]._out[0].label = q-k-p
        n[3].link_vertex(n[4])
        n[3]._out[0].label = q-k-p-r
        n[4].link_vertex(n[1])
        n[4]._out[0].label = -k
        n[2].add_outgoing()
        n[2]._out[1].label = p
        n[3].add_outgoing()
        n[3]._out[1].label = r
        n[4].add_outgoing()
        n[4]._out[1].label = q-p-r
        h = restflow.Graph(n)
        h.vertices[0]._in[0].label = q

        for i in range(len(g.vertices)):
            for j in range(len(g.vertices[i]._in)):
                self.assertEqual((g.vertices[i]._in[j].label)**2, (h.vertices[i]._in[j].label)**2)

    def test_freq_integral(self):
        model = AMBp()
        #graph k
        v = [restflow.Vertex() for i in range(5)]
        v[0].link_vertex(v[1])
        v[0].link_vertex(v[2])
        v[2].link_vertex(v[3])
        v[3].link_vertex(v[4])
        v[4].link_vertex(v[1])
        v[2].add_outgoing()
        v[3].add_outgoing()
        v[4].add_outgoing()
        g1 = restflow.Graph(v)    
        labels = [k, p, r, q-p-r]
        g1.label_edges(labels)
        Q = 1/((model.f(k)+model.f(q-k))*(model.f(k)+model.f(q-p-k))*(model.f(k)+model.f(q-p-k-r)))
        self.assertEqual(g1._calculate_freq_integral(g1.k,model.f)[0]/g1._calculate_freq_integral(g1.k,model.f)[1], Q)

        #graph j
        v = [restflow.Vertex() for i in range(5)]
        v[0].link_vertex(v[1])
        v[1].link_vertex(v[2])
        v[2].link_vertex(v[3])
        v[0].link_vertex(v[4])
        v[4].link_vertex(v[3])
        v[1].add_outgoing()
        v[2].add_outgoing()
        v[4].add_outgoing()
        g2 = restflow.Graph(v)    
        labels = [k, p, r, q-p-r]
        g2.label_edges(labels)
        Qppm =(2*model.f(k)*(model.f(k)+model.f(p+r+k)+model.f(r+k)+model.f(q-p-r-k))+model.f(q-p-r-k)**2+model.f(p+r+k)*model.f(r+k)+model.f(p+r+k)*model.f(q-p-r-k)+model.f(r+k)*model.f(q-p-r-k))/((model.f(p+r+k)+model.f(q-p-r-k))*(model.f(r+k)+model.f(q-p-r-k)))
        Q = 1/((model.f(k)+model.f(p+r+k))*(model.f(k)+model.f(r+k))*(model.f(k)+model.f(q-p-r-k)))*Qppm
        self.assertEqual(g2._calculate_freq_integral(g2.k,model.f)[0]/g2._calculate_freq_integral(g2.k,model.f)[1], Q)
        
        #graph b
        v = [restflow.Vertex() for i in range(4)]
        v[0].link_vertex(v[1])
        v[0].link_vertex(v[2])
        v[2].link_vertex(v[3])
        v[3].link_vertex(v[1])
        v[2].add_outgoing()
        v[3].add_outgoing()
        g3 = restflow.Graph(v)
        labels = [k, p, q-p]
        g3.label_edges(labels)
        Q = 1/((model.f(k)+model.f(q-k))*(model.f(k)+model.f(q-p-k)))
        self.assertEqual(g3._calculate_freq_integral(g3.k,model.f)[0]/g3._calculate_freq_integral(g3.k,model.f)[1], Q) 

        #graph a
        v = [restflow.Vertex() for i in range(4)]
        v[0].link_vertex(v[1])
        v[0].link_vertex(v[2])
        v[1].link_vertex(v[3])
        v[2].link_vertex(v[3])
        v[1].add_outgoing()
        v[2].add_outgoing()# example: Figure (a) from above graph
        g4 = restflow.Graph(v)
        labels = [k, p, q-p]
        g4.label_edges(labels)
        Q = 1/((model.f(k)+model.f(p+k))*(model.f(k)+model.f(q-p-k)))*(2*model.f(k)+model.f(p+k)+model.f(q-p-k))/(model.f(p+k)+model.f(q-p-k))
        self.assertEqual(g4._calculate_freq_integral(g4.k,model.f)[0]/g4._calculate_freq_integral(g4.k,model.f)[1], Q)

        #kpz diagram
        v = [restflow.Vertex() for i in range(3)]
        v[0].link_vertex(v[1])
        v[0].link_vertex(v[2])
        v[1].link_vertex(v[2])
        v[1].add_outgoing()
        g5 = restflow.Graph(v)
        labels = [k, q]
        g5.label_edges(labels)
        Q = 1/(model.f(q-k)+model.f(k))
        self.assertEqual(g5._calculate_freq_integral(g5.k,model.f)[0]/g5._calculate_freq_integral(g5.k,model.f)[1], Q)

    def test_multiplicity(self):
        #graph d
        v = [restflow.Vertex() for i in range(3)]
        v[0].link_vertex(v[1])
        v[0].add_outgoing()
        v[0].link_vertex(v[2])
        v[2].link_vertex(v[1])
        v[2].add_outgoing()
        g1 = restflow.Graph(v)
        self.assertEqual(g1.calculate_multiplicity(), 12)

        #graph j
        v = [restflow.Vertex() for i in range(5)]
        v[0].link_vertex(v[1])
        v[1].link_vertex(v[2])
        v[2].link_vertex(v[3])
        v[0].link_vertex(v[4])
        v[4].link_vertex(v[3])
        v[1].add_outgoing()
        v[2].add_outgoing()
        v[4].add_outgoing()
        g2 = restflow.Graph(v)
        self.assertEqual(g2.calculate_multiplicity(), 8)

        v = [restflow.Vertex() for i in range(4)]
        v[0].link_vertex(v[1])
        v[0].link_vertex(v[2])
        v[2].link_vertex(v[3])
        v[3].link_vertex(v[1])
        v[0].add_outgoing()
        v[2].add_outgoing()
        v[3].add_outgoing()
        g3 = restflow.Graph(v)
        self.assertEqual(g3.calculate_multiplicity(), 24)

    def test_integral(self):
        model = AMBp()
        #kpz
        v = [restflow.Vertex() for i in range(3)]
        v[0].link_vertex(v[1])
        v[0].link_vertex(v[2])
        v[1].link_vertex(v[2])
        v[1].add_outgoing()
        g1 = restflow.Graph(v)
        labels = [k, q]
        g1.label_edges(labels)
        nom, den = g1.convert(model).num, g1.convert(model).den
        Itruth =  model.D*g1.k**model.alpha*g1.calculate_multiplicity()*g1._calculate_freq_integral(g1.k,model.f)[0]*model.v2(k,q-k,q)[0]*model.v2(-k,q,q-k)[0]/(model.f(g1.k)*g1._calculate_freq_integral(g1.k,model.f)[1]*model.v2(k,q-k,q)[1]*model.v2(-k,q,q-k)[1])
        self.assertEqual(nom/den, Itruth)

        #figure c
        v = [restflow.Vertex() for i in range(3)]
        v[0].link_vertex(v[1])
        v[0].link_vertex(v[2])
        v[2].link_vertex(v[1])
        v[2].add_outgoing()
        v[2].add_outgoing()
        g2 = restflow.Graph(v)
        labels = [k, q-p, p]
        g2.label_edges(labels)
        nom, den = g2.convert(model).num, g2.convert(model).den
        Itruth = model.D*g2.k**model.alpha*  g2.calculate_multiplicity()*g2._calculate_freq_integral(g2.k,model.f)[0]*model.v2(k,q-k,q)[0]*model.v3(-k,p,q-p,q-k)[0]/(model.f(g2.k)*g2._calculate_freq_integral(g2.k,model.f)[1]*model.v2(k,q-k,q)[1]*model.v3(-k,p,q-p,q-k)[1])
        self.assertEqual(nom/den, Itruth)

        #figure 3c
        v = [restflow.Vertex() for i in range(3)]
        v[0].link_vertex(v[1])
        v[0].link_vertex(v[2])
        v[2].link_vertex(v[1])
        v[0].add_outgoing()
        v[2].add_outgoing()
        v[2].add_outgoing()
        g3 = restflow.Graph(v)
        labels = [k, p, r, q-p-r]
        g3.label_edges(labels)
        nom, den = g3.convert(model).num, g3.convert(model).den
        Itruth = model.D*g3.k**model.alpha*g3.calculate_multiplicity()*g3._calculate_freq_integral(g3.k,model.f)[0]*model.v3(k,p,q-p-k,q)[0]*model.v3(r,q-p-r,-k,q-p-k)[0]/(model.f(g3.k)*g3._calculate_freq_integral(g3.k,model.f)[1]*model.v3(k,p,q-p-k,q)[1]*model.v3(r,q-p-r,-k,q-p-k)[1])
        self.assertEqual(nom/den, Itruth)

    def test_symmetrize(self):
        _q, _k, _p, dot_kq, dot_pk, dot_qp = sympy.symbols('q k 0 (k·q) 0 0')
        _r, dot_qr, dot_pr, dot_rk = sympy.symbols('0 0 0 0')
        ctx = symvec.Context()
        ctx.add_dot_product(_q,_k,dot_kq)
        ctx.add_dot_product(_q,_p,dot_qp)
        ctx.add_dot_product(_p,_k,dot_pk)
        ctx.add_dot_product(_q,_r,dot_qr)
        ctx.add_dot_product(_p,_r,dot_pr)
        ctx.add_dot_product(_r,_k,dot_rk)
        k = ctx.vector(_k)
        q = ctx.vector(_q)
        p = ctx.vector(_p)
        model = AMBp()

        # graph (d)
        v = [restflow.Vertex() for i in range(3)]
        v[0].link_vertex(v[1])
        v[0].link_vertex(v[2])
        v[2].link_vertex(v[1])
        v[0].add_outgoing()
        v[2].add_outgoing()
        g = restflow.Graph(v)
        exprs = g.convert_perm(model,[k,p,q-p])
        Ic = restflow.integrate(exprs,3,[k,p,q-p])
        labels = [k, p, q-p]
        g.label_edges(labels)
        expr = g.convert(model)
        Ic1 = restflow.integrate([expr],3,labels)
        labels = [k, q-p, p]
        g.label_edges(labels)
        expr = g.convert(model)
        Ic2 = restflow.integrate([expr],3,labels)
        Ictotal= sympy.simplify(sympy.Poly((Ic1+Ic2)/2,q.sym).as_expr())
        self.assertEqual(Ic, Ictotal)

if __name__ == '__main__':
    unittest.main(verbosity=2)
