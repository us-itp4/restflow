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
#parameters of model
c1, c2, c3, u, D, kappa = sympy.symbols('c1 c2 c3 u D kappa')
c1t, c2t, c3t, ut, Dt, kappat = sympy.symbols('c1t c2t c3t ut Dt kappat')
# dimension and reduced surface area
d, Kd = sympy.symbols('d K_d')

# symbols for vectors
_q, _k, dot_kq = sympy.symbols('q k (k·q)')
_p, _k, dot_pk = sympy.symbols('p k (k·p)')
_q, _p, dot_qp = sympy.symbols('q p (q·p)')
_q, _r, dot_qr = sympy.symbols('q r (q·r)')
_p, _r, dot_pr = sympy.symbols('p r (p·r)')
_r, _k, dot_rk = sympy.symbols('r k (r·k)')

ctx = symvec.Context()
ctx.add_dot(_q, _k, dot_kq)
ctx.add_dot(_p, _k, dot_pk)
ctx.add_dot(_q, _p, dot_qp)
ctx.add_dot(_q, _r, dot_qr)
ctx.add_dot(_p, _r, dot_pr)
ctx.add_dot(_r, _k, dot_rk)

# create vectors
k = ctx.vector(_k)
q = ctx.vector(_q)
p = ctx.vector(_p)
r = ctx.vector(_r)

class AMBp:
    def __init__(self):
        self.alpha = 2
        self.c1, self.c2, self.c3, self.u, self.D, self.kappa = sympy.symbols('c1 c2 c3 u D kappa')

    def f(self,k):
        return (self.kappa*k**2)

    def v2(self,k1,k2,q):
        expr = (self.c1*q**2+2*self.c2*k1*k2-self.c3*(k2**2*q*k1+k1**2*q*k2)/(q**2))*1/2
        return sympy.fraction(sympy.cancel((expr)))

    def v3(self,k1,k2,k3,q):
        return (-self.u,1)

model = AMBp()
'''------------------------------------------------------------------------------------------------------------'''

def test_integrate2():
    assert symvec.integrate2((model.c1*q**2,sympy.sympify(2)), k, q, ctx, d, n=5) == model.c1*q**2/2
    assert symvec.integrate2((dot_kq**2*model.c3,q**2), k, q, ctx, d, n=5) == model.c3*k**2/d
    assert symvec.integrate2((dot_kq*model.c2,sympy.sympify(1)), k, q, ctx, d, n=5) == 0
    assert symvec.integrate2(((1+dot_kq)*(2+dot_kq),sympy.sympify(1)), k, q, ctx, d, n=5) == 2+k**2*q**2/d
    assert symvec.integrate2((sympy.sympify(1),(k+q)**2), k, q, ctx, d, n=5) == 1/k.sym**2 + q**2*(-1 + 4/d)/k.sym**4 + q**4*(1 - 12/d + 48/(d*(d + 2)))/k.sym**6

def test_integrate3():
    cs_psi = sympy.symbols('cos_psi')
    assert symvec.integrate3(((q-p-k)*(q-k),sympy.sympify(1)), k, q, p, ctx, d, n=5) == -cs_psi*p.sym*q.sym+k**2+q**2
    assert symvec.integrate3((((k)*(q+p))**2,sympy.sympify(1)), k, q, p, ctx, d, n=5) == 2*cs_psi*k**2*p.sym*q.sym/d + k**2*p**2/d + k**2*q**2/d
    assert symvec.integrate3((sympy.sympify(1),(q+k+p)**2), k, q, p, ctx, d, n=3) == k.sym**(-2) + p**2*(4 - d)/(d*k.sym**4) + p.sym*q.sym*(-2*cs_psi*d + 8*cs_psi)/(d*k.sym**4) + q**2*(4 - d)/(d*k.sym**4) 

def test_label_edges():
    #kpz figure
    v = [restflow.Vertex() for i in range(3)]
    v[0].link_vertex(v[1],0)
    v[0].link_vertex(v[2],0)
    v[1].link_vertex(v[2],0)
    v[1].add_outgoing(0)
    g = restflow.Graph(v)
    g.label_edges(k,[q])

    n = [restflow.Vertex() for i in range(3)]
    n[0].link_vertex(n[1],0)
    n[0]._out[0].label = q-k
    n[0].link_vertex(n[2],0)
    n[0]._out[1].label = k
    n[1].link_vertex(n[2],0)
    n[1]._out[0].label = -k
    n[1].add_outgoing(0)
    n[1]._out[1].label = q
    h = restflow.Graph(n)
    h.vertices[0]._in[0].label=q

    for i in range(len(g.vertices)):
        for j in range(len(g.vertices[i]._in)):
            assert (g.vertices[i]._in[j].label)**2 == (h.vertices[i]._in[j].label)**2

    v = [restflow.Vertex() for i in range(5)]
    v[0].link_vertex(v[1],0)
    v[0].link_vertex(v[2],0)
    v[2].link_vertex(v[3],0)
    v[3].link_vertex(v[4],0)
    v[4].link_vertex(v[1],0)
    v[2].add_outgoing(0)
    v[3].add_outgoing(0)
    v[4].add_outgoing(0)
    g = restflow.Graph(v)
    g.label_edges(k,[p,r,q-p-r])

    n = [restflow.Vertex() for i in range(5)]
    n[0].link_vertex(n[1],0)
    n[0]._out[0].label = k
    n[0].link_vertex(n[2],0)
    n[0]._out[1].label = q-k
    n[2].link_vertex(n[3],0)
    n[2]._out[0].label = q-k-p
    n[3].link_vertex(n[4],0)
    n[3]._out[0].label = q-k-p-r
    n[4].link_vertex(n[1],0)
    n[4]._out[0].label = -k
    n[2].add_outgoing(0)
    n[2]._out[1].label = p
    n[3].add_outgoing(0)
    n[3]._out[1].label = r
    n[4].add_outgoing(0)
    n[4]._out[1].label = q-p-r
    h = restflow.Graph(n)
    h.vertices[0]._in[0].label = q

    for i in range(len(g.vertices)):
        for j in range(len(g.vertices[i]._in)):
            assert (g.vertices[i]._in[j].label)**2 == (h.vertices[i]._in[j].label)**2

def test_freq_integral():
    #graph k
    v = [restflow.Vertex() for i in range(5)]
    v[0].link_vertex(v[1],0)
    v[0].link_vertex(v[2],0)
    v[2].link_vertex(v[3],0)
    v[3].link_vertex(v[4],0)
    v[4].link_vertex(v[1],0)
    v[2].add_outgoing(0)
    v[3].add_outgoing(0)
    v[4].add_outgoing(0)
    g1 = restflow.Graph(v)    
    g1.label_edges(k,[p,r,q-p-r])
    assert g1._calculate_freq_integral(g1.k,model.f)[0]/g1._calculate_freq_integral(g1.k,model.f)[1] == 1/((model.f(k)+model.f(q-k))*(model.f(k)+model.f(q-p-k))*(model.f(k)+model.f(q-p-k-r)))

    #graph j
    v = [restflow.Vertex() for i in range(5)]
    v[0].link_vertex(v[1],0)
    v[1].link_vertex(v[2],0)
    v[2].link_vertex(v[3],0)
    v[0].link_vertex(v[4],0)
    v[4].link_vertex(v[3],0)
    v[1].add_outgoing(0)
    v[2].add_outgoing(0)
    v[4].add_outgoing(0)
    g2 = restflow.Graph(v)    
    g2.label_edges(k,[p,r,q-p-r])
    Qppm=(2*model.f(k)*(model.f(k)+model.f(p+r+k)+model.f(r+k)+model.f(q-p-r-k))+model.f(q-p-r-k)**2+model.f(p+r+k)*model.f(r+k)+model.f(p+r+k)*model.f(q-p-r-k)+model.f(r+k)*model.f(q-p-r-k))/((model.f(p+r+k)+model.f(q-p-r-k))*(model.f(r+k)+model.f(q-p-r-k)))
    assert g2._calculate_freq_integral(g2.k,model.f)[0]/g2._calculate_freq_integral(g2.k,model.f)[1] == 1/((model.f(k)+model.f(p+r+k))*(model.f(k)+model.f(r+k))*(model.f(k)+model.f(q-p-r-k)))*Qppm
    
    #graph b
    v = [restflow.Vertex() for i in range(4)]
    v[0].link_vertex(v[1],0)
    v[0].link_vertex(v[2],0)
    v[2].link_vertex(v[3],0)
    v[3].link_vertex(v[1],0)
    v[2].add_outgoing(0)
    v[3].add_outgoing(0)
    g3 = restflow.Graph(v)
    g3.label_edges(k,[p,q-p])
    assert g3._calculate_freq_integral(g3.k,model.f)[0]/g3._calculate_freq_integral(g3.k,model.f)[1] == 1/((model.f(k)+model.f(q-k))*(model.f(k)+model.f(q-p-k))) 

    #graph a
    v = [restflow.Vertex() for i in range(4)]
    v[0].link_vertex(v[1],0)
    v[0].link_vertex(v[2],0)
    v[1].link_vertex(v[3],0)
    v[2].link_vertex(v[3],0)
    v[1].add_outgoing(0)
    v[2].add_outgoing(0)# example: Figure (a) from above graph
    g4 = restflow.Graph(v)
    g4.label_edges(k,[p,q-p])
    assert g4._calculate_freq_integral(g4.k,model.f)[0]/g4._calculate_freq_integral(g4.k,model.f)[1] == 1/((model.f(k)+model.f(p+k))*(model.f(k)+model.f(q-p-k)))*(2*model.f(k)+model.f(p+k)+model.f(q-p-k))/(model.f(p+k)+model.f(q-p-k))

    #kpz diagram
    v = [restflow.Vertex() for i in range(3)]
    v[0].link_vertex(v[1],0)
    v[0].link_vertex(v[2],0)
    v[1].link_vertex(v[2],0)
    v[1].add_outgoing(0)
    g5 = restflow.Graph(v)
    g5.label_edges(k,[q])
    assert g5._calculate_freq_integral(g5.k,model.f)[0]/g5._calculate_freq_integral(g5.k,model.f)[1] == 1/(model.f(q-k)+model.f(k))

def test_multiplicity():
    #graph d
    v = [restflow.Vertex() for i in range(3)]
    v[0].link_vertex(v[1],0)
    v[0].add_outgoing(0)
    v[0].link_vertex(v[2],0)
    v[2].link_vertex(v[1],0)
    v[2].add_outgoing(0)
    g1 = restflow.Graph(v)
    assert g1._calculate_multiplicity() == 12

    #graph j
    v = [restflow.Vertex() for i in range(5)]
    v[0].link_vertex(v[1],0)
    v[1].link_vertex(v[2],0)
    v[2].link_vertex(v[3],0)
    v[0].link_vertex(v[4],0)
    v[4].link_vertex(v[3],0)
    v[1].add_outgoing(0)
    v[2].add_outgoing(0)
    v[4].add_outgoing(0)
    g2 = restflow.Graph(v)
    assert g2._calculate_multiplicity() == 8

    v = [restflow.Vertex() for i in range(4)]
    v[0].link_vertex(v[1],0)
    v[0].link_vertex(v[2],0)
    v[2].link_vertex(v[3],0)
    v[3].link_vertex(v[1],0)
    v[0].add_outgoing(0)
    v[2].add_outgoing(0)
    v[3].add_outgoing(0)
    g3 = restflow.Graph(v)
    assert g3._calculate_multiplicity() == 24

def test_integral():
    s0 =  D/model.f(k)
    #kpz
    v = [restflow.Vertex() for i in range(3)]
    v[0].link_vertex(v[1],0)
    v[0].link_vertex(v[2],0)
    v[1].link_vertex(v[2],0)
    v[1].add_outgoing(0)
    g1 = restflow.Graph(v)
    g1.label_edges(k,[q])
    nom, den = g1.convert(model)
    assert nom/den ==  model.D*g1.k**model.alpha*g1._calculate_multiplicity()*g1._calculate_freq_integral(g1.k,model.f)[0]*model.v2(k,q-k,q)[0]*model.v2(-k,q,q-k)[0]/(model.f(g1.k)*g1._calculate_freq_integral(g1.k,model.f)[1]*model.v2(k,q-k,q)[1]*model.v2(-k,q,q-k)[1])

    #figure c
    v = [restflow.Vertex() for i in range(3)]
    v[0].link_vertex(v[1],0)
    v[0].link_vertex(v[2],0)
    v[2].link_vertex(v[1],0)
    v[2].add_outgoing(0)
    v[2].add_outgoing(0)
    g2 = restflow.Graph(v)
    g2.label_edges(k,[q-p,p])
    nom, den = g2.convert(model)
    assert nom/den == model.D*g2.k**model.alpha*  g2._calculate_multiplicity()*g2._calculate_freq_integral(g2.k,model.f)[0]*model.v2(k,q-k,q)[0]*model.v3(-k,p,q-p,q-k)[0]/(model.f(g2.k)*g2._calculate_freq_integral(g2.k,model.f)[1]*model.v2(k,q-k,q)[1]*model.v3(-k,p,q-p,q-k)[1])

    #figure 3c
    v = [restflow.Vertex() for i in range(3)]
    v[0].link_vertex(v[1],0)
    v[0].link_vertex(v[2],0)
    v[2].link_vertex(v[1],0)
    v[0].add_outgoing(0)
    v[2].add_outgoing(0)
    v[2].add_outgoing(0)
    g3 = restflow.Graph(v)
    g3.label_edges(k,[p,r,q-p-r])
    nom, den = g3.convert(model)
    assert nom/den == model.D*g3.k**model.alpha*g3._calculate_multiplicity()*g3._calculate_freq_integral(g3.k,model.f)[0]*model.v3(k,p,q-p-k,q)[0]*model.v3(r,q-p-r,-k,q-p-k)[0]/(model.f(g3.k)*g3._calculate_freq_integral(g3.k,model.f)[1]*model.v3(k,p,q-p-k,q)[1]*model.v3(r,q-p-r,-k,q-p-k)[1])

def test_symmetrize():
    s0 =  D/model.f(k)
    # figure j
    v = [restflow.Vertex() for i in range(5)]
    v[0].link_vertex(v[1],0)
    v[1].link_vertex(v[2],0)
    v[2].link_vertex(v[3],0)
    v[0].link_vertex(v[4],0)
    v[4].link_vertex(v[3],0)
    v[1].add_outgoing(0)
    v[2].add_outgoing(0)
    v[4].add_outgoing(0)
    I_array = []
    I_array = restflow.integrals.symmetrize(v, q, [p,r,q-r-p], k, model)
    I_array = [(element[0].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)]), element[1].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)])) for element in I_array]

    g1 = restflow.Graph(v)
    g1.label_edges(k,[p,r,q-p-r])
    nom, den = g1.convert(model)
    I1 = (nom,den*int(6))
    I1 = (I1[0].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)]),I1[1].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)]))
    assert I_array[0] == I1

    v = [restflow.Vertex() for i in range(5)]
    v[0].link_vertex(v[1],0)
    v[1].link_vertex(v[2],0)
    v[2].link_vertex(v[3],0)
    v[0].link_vertex(v[4],0)
    v[4].link_vertex(v[3],0)
    v[1].add_outgoing(0)
    v[2].add_outgoing(0)
    v[4].add_outgoing(0)
    g2 = restflow.Graph(v)
    g2.label_edges(k,[p,q-p-r,r])
    nom, den = g2.convert(model)
    I2 = (nom,den*int(6))
    I2 = (I2[0].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)]), I2[1].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)]))
    assert I_array[1] == I2

    v = [restflow.Vertex() for i in range(5)]
    v[0].link_vertex(v[1],0)
    v[1].link_vertex(v[2],0)
    v[2].link_vertex(v[3],0)
    v[0].link_vertex(v[4],0)
    v[4].link_vertex(v[3],0)
    v[1].add_outgoing(0)
    v[2].add_outgoing(0)
    v[4].add_outgoing(0)
    g3 = restflow.Graph(v)
    g3.label_edges(k,[q-p-r,p,r])
    nom, den = g3.convert(model)
    I3 = (nom,den*int(6))
    I3 = (I3[0].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)]), I3[1].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)]))
    assert I_array[4] == I3

test_integrate2()
test_integrate3()
test_label_edges()
test_freq_integral()
test_multiplicity()
test_integral()
test_symmetrize()
print('All tests passed succesfully!')