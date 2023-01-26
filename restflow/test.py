from graph import *
# from graph_original import *
import sympy
import symvec
from IPython.display import display
from sympy import *

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

# assign symbol for dot product
symvec.dots[frozenset((_q,_k))] = dot_kq
symvec.dots[frozenset((_q,_p))] = dot_qp
symvec.dots[frozenset((_p,_k))] = dot_pk
symvec.dots[frozenset((_q,_r))] = dot_qr
symvec.dots[frozenset((_p,_r))] = dot_pr
symvec.dots[frozenset((_r,_k))] = dot_rk
# create vectors
k = symvec.Vector(_k)
q = symvec.Vector(_q)
p = symvec.Vector(_p)
r = symvec.Vector(_r)
def f(x, alpha, a):
    return (kappa*x**2+a)*x**alpha
# vertex functions
# IMPORTANT: Simplify the vertex at the output
def v2(k1,k2,k3):
    expr = (c1*k3**2+2*c2*k1*k2-c3*(k2**2*k3*k1+k1**2*k3*k2)/(k3**2))*1/2
    return fraction(cancel((expr)))

def v2t(k1,k2,k3):
    # renormalized v2 (model parameters with tilda)
    return simplify((c1t*k3**2+2*c2t*k1*k2-c3t*(k2**2*k3*k1+k1**2*k3*k2)/(k3**2))*1/2)

def v3(k1,k2,k3,k4):
    return (-u,1)

'''------------------------------------------------------------------------------------------------------------'''

def test_integrate2():
    assert symvec.integrate2((c1*q**2,sympify(2)), k, q, d, n=5) == c1*q**2/2
    assert symvec.integrate2((dot_kq**2*c3,q**2), k, q, d, n=5) == c3*k**2/d
    assert symvec.integrate2((dot_kq*c2,sympify(1)), k, q, d, n=5) == 0
    assert symvec.integrate2(((1+dot_kq)*(2+dot_kq),sympify(1)), k, q, d, n=5) == 2+k**2*q**2/d
    assert symvec.integrate2((sympify(1),(k+q)**2), k, q, d, n=5) == 1/k.sym**2 + q**2*(-1 + 4/d)/k.sym**4 + q**4*(1 - 12/d + 48/(d*(d + 2)))/k.sym**6

def test_integrate3():
    cs_psi = sympy.symbols('cos_psi')
    assert symvec.integrate3(((q-p-k)*(q-k),sympify(1)), k, q, p, d, n=5) == -cs_psi*p.sym*q.sym+k**2+q**2
    assert symvec.integrate3((((k)*(q+p))**2,sympify(1)), k, q, p, d, n=5) == 2*cs_psi*k**2*p.sym*q.sym/d + k**2*p**2/d + k**2*q**2/d
    assert symvec.integrate3((sympify(1),(q+k+p)**2), k, q, p, d, n=3) == k.sym**(-2) + p**2*(4 - d)/(d*k.sym**4) + p.sym*q.sym*(-2*cs_psi*d + 8*cs_psi)/(d*k.sym**4) + q**2*(4 - d)/(d*k.sym**4) 

def test_label_edges():
    #kpz figure
    v = [Vertex() for i in range(3)]
    v[0].link_vertex(v[1])
    v[0].link_vertex(v[2])
    v[1].link_vertex(v[2])
    v[1].add_outgoing()
    g = Graph(v)
    g.label_edges(q,[q],k)

    n = [Vertex() for i in range(3)]
    n[0].link_vertex(n[1])
    n[0]._out[0].label = q-k
    n[0].link_vertex(n[2])
    n[0]._out[1].label = k
    n[1].link_vertex(n[2])
    n[1]._out[0].label = -k
    n[1].add_outgoing()
    n[1]._out[1].label = q
    h = Graph(n)
    h._root_(q)

    for i in range(len(g.coeff)):
        for j in range(len(g.coeff[i]._in)):
            assert (g.coeff[i]._in[j].label)**2 == (h.coeff[i]._in[j].label)**2

    v = [Vertex() for i in range(5)]
    v[0].link_vertex(v[1])
    v[0].link_vertex(v[2])
    v[2].link_vertex(v[3])
    v[3].link_vertex(v[4])
    v[4].link_vertex(v[1])
    v[2].add_outgoing()
    v[3].add_outgoing()
    v[4].add_outgoing()
    g = Graph(v)
    g.label_edges(q,[p,r,q-p-r],k)

    n = [Vertex() for i in range(5)]
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
    h = Graph(v)

    for i in range(len(g.coeff)):
        for j in range(len(g.coeff[i]._in)):
            assert (g.coeff[i]._in[j].label)**2 == (h.coeff[i]._in[j].label)**2

def test_freq_integral():
    alpha, a = 2, 0
    s0 =  D/f(k, alpha, a)
    #graph k
    v = [Vertex() for i in range(5)]
    v[0].link_vertex(v[1])
    v[0].link_vertex(v[2])
    v[2].link_vertex(v[3])
    v[3].link_vertex(v[4])
    v[4].link_vertex(v[1])
    v[2].add_outgoing()
    v[3].add_outgoing()
    v[4].add_outgoing()
    g1 = Graph(v)    
    # assert g1.freq_integral(f,D,k,alpha,a) == s0*k**2/((f(k, alpha, a)+f(q-k, alpha, a))*(f(k, alpha, a)+f(q-p-k, alpha, a))*(f(k, alpha, a)+f(q-p-k-r, alpha, a)))

    #graph j
    v = [Vertex() for i in range(5)]
    v[0].link_vertex(v[1])
    v[1].link_vertex(v[2])
    v[2].link_vertex(v[3])
    v[0].link_vertex(v[4])
    v[4].link_vertex(v[3])
    v[1].add_outgoing()
    v[2].add_outgoing()
    v[4].add_outgoing()
    g2 = Graph(v)    
    g2.label_edges(q,[p,r,q-p-r],k)
    Qppm=(2*f(k,alpha, a)*(f(k,alpha, a)+f(p+r+k,alpha,a)+f(r+k,alpha,a)+f(q-p-r-k,alpha,a))+f(q-p-r-k,alpha,a)**2+f(p+r+k,alpha,a)*f(r+k,alpha,a)+f(p+r+k,alpha,a)*f(q-p-r-k,alpha,a)+f(r+k,alpha,a)*f(q-p-r-k,alpha,a))/((f(p+r+k,alpha,a)+f(q-p-r-k,alpha,a))*(f(r+k,alpha,a)+f(q-p-r-k,alpha,a)))
    assert g2.freq_integral(f,D,k,alpha,a)[0]/g2.freq_integral(f,D,k,alpha,a)[1] == s0*k**2/((f(k,alpha,a)+f(p+r+k,alpha,a))*(f(k,alpha,a)+f(r+k,alpha,a))*(f(k,alpha,a)+f(q-p-r-k,alpha,a)))*Qppm
    #graph b
    v = [Vertex() for i in range(4)]
    v[0].link_vertex(v[1])
    v[0].link_vertex(v[2])
    v[2].link_vertex(v[3])
    v[3].link_vertex(v[1])
    v[2].add_outgoing()
    v[3].add_outgoing()
    g3 = Graph(v)
    g3.label_edges(q,[p,q-p],k)
    assert g3.freq_integral(f,D,k,alpha,a)[0]/g3.freq_integral(f,D,k,alpha,a)[1] == s0*k**2/((f(k,alpha,a)+f(q-k,alpha,a))*(f(k,alpha,a)+f(q-p-k,alpha,a))) 

    #graph a
    v = [Vertex() for i in range(4)]
    v[0].link_vertex(v[1])
    v[0].link_vertex(v[2])
    v[1].link_vertex(v[3])
    v[2].link_vertex(v[3])
    v[1].add_outgoing()
    v[2].add_outgoing()# example: Figure (a) from above graph
    g4 = Graph(v)
    g4.label_edges(q,[p,q-p],k)
    assert g4.freq_integral(f,D,k,alpha,a)[0]/g4.freq_integral(f,D,k,alpha,a)[1] == s0*k**2/((f(k,alpha,a)+f(p+k,alpha,a))*(f(k,alpha,a)+f(q-p-k,alpha,a)))*(2*f(k,alpha,a)+f(p+k,alpha,a)+f(q-p-k,alpha,a))/(f(p+k,alpha,a)+f(q-p-k,alpha,a))

    #kpz diagram
    v = [Vertex() for i in range(3)]
    v[0].link_vertex(v[1])
    v[0].link_vertex(v[2])
    v[1].link_vertex(v[2])
    v[1].add_outgoing()
    g5 = Graph(v)
    g5.label_edges(q,[q],k)
    assert g5.freq_integral(f,D,k,alpha,a)[0]/g5.freq_integral(f,D,k,alpha,a)[1] == s0*k**2/(f(q-k,alpha,a)+f(k,alpha,a))

def test_multiplicity():
    #graph d
    v = [Vertex() for i in range(3)]
    v[0].link_vertex(v[1])
    v[0].add_outgoing()
    v[0].link_vertex(v[2])
    v[2].link_vertex(v[1])
    v[2].add_outgoing()
    g1 = Graph(v)
    assert g1.multiplicity() == 12

    #graph j
    v = [Vertex() for i in range(5)]
    v[0].link_vertex(v[1])
    v[1].link_vertex(v[2])
    v[2].link_vertex(v[3])
    v[0].link_vertex(v[4])
    v[4].link_vertex(v[3])
    v[1].add_outgoing()
    v[2].add_outgoing()
    v[4].add_outgoing()
    g2 = Graph(v)
    assert g2.multiplicity() == 8

    v = [Vertex() for i in range(4)]
    v[0].link_vertex(v[1])
    v[0].link_vertex(v[2])
    v[2].link_vertex(v[3])
    v[3].link_vertex(v[1])
    v[0].add_outgoing()
    v[2].add_outgoing()
    v[3].add_outgoing()
    g3 = Graph(v)
    assert g3.multiplicity() == 24

def test_integral():
    alpha, a = 2, 0
    s0 =  D/f(k, alpha, a)
    #kpz
    v = [Vertex() for i in range(3)]
    v[0].link_vertex(v[1])
    v[0].link_vertex(v[2])
    v[1].link_vertex(v[2])
    v[1].add_outgoing()
    g1 = Graph(v)
    g1.label_edges(q,[q],k)
    assert g1.integral(f, D, k, alpha, a, v2, v3)[0]/g1.integral(f, D, k, alpha, a, v2, v3)[1] ==  g1.multiplicity()*g1.freq_integral(f, D,k,alpha,a)[0]*v2(k,q-k,q)[0]*v2(-k,q,q-k)[0]/(g1.freq_integral(f, D,k,alpha,a)[1]*v2(k,q-k,q)[1]*v2(-k,q,q-k)[1])

    #figure c
    v = [Vertex() for i in range(3)]
    v[0].link_vertex(v[1])
    v[0].link_vertex(v[2])
    v[2].link_vertex(v[1])
    v[2].add_outgoing()
    v[2].add_outgoing()
    g2 = Graph(v)
    g2.label_edges(q,[q-p,p],k)
    assert g2.integral(f, D, k, alpha, a, v2, v3)[0]/g2.integral(f, D, k, alpha, a, v2, v3)[1] ==  g2.multiplicity()*g2.freq_integral(f, D,k,alpha,a)[0]*v2(k,q-k,q)[0]*v3(-k,p,q-p,q-k)[0]/(g2.freq_integral(f, D,k,alpha,a)[1]*v2(k,q-k,q)[1]*v3(-k,p,q-p,q-k)[1])

    #figure 3c
    v = [Vertex() for i in range(3)]
    v[0].link_vertex(v[1])
    v[0].link_vertex(v[2])
    v[2].link_vertex(v[1])
    v[0].add_outgoing()
    v[2].add_outgoing()
    v[2].add_outgoing()
    g3 = Graph(v)
    g3.label_edges(q,[p,r,q-p-r],k)
    assert g3.integral(f, D, k, alpha, a, v2, v3)[0]/g3.integral(f, D, k, alpha, a, v2, v3)[1] ==  g3.multiplicity()*g3.freq_integral(f,D,k,alpha,a)[0]*v3(k,p,q-p-k,q)[0]*v3(r,q-p-r,-k,q-p-k)[0]/(g3.freq_integral(f,D,k,alpha,a)[1]*v3(k,p,q-p-k,q)[1]*v3(r,q-p-r,-k,q-p-k)[1])

def test_integrals_symmet():
    alpha, a = 2, 0
    s0 =  D/f(k, alpha, a)
    # figure j
    v = [Vertex() for i in range(5)]
    v[0].link_vertex(v[1])
    v[1].link_vertex(v[2])
    v[2].link_vertex(v[3])
    v[0].link_vertex(v[4])
    v[4].link_vertex(v[3])
    v[1].add_outgoing()
    v[2].add_outgoing()
    v[4].add_outgoing()
    I_array = []
    I_array = integrals_symmet(v, q, [p,r,q-r-p], f, D, k, alpha, a, v2, v3)
    I_array = [(element[0].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)]), element[1].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)])) for element in I_array]

    g1 = Graph(v)
    g1.label_edges(q,[p,r,q-p-r],k)
    I1 = (g1.integral(f, D, k, alpha, a, v2, v3)[0],g1.integral(f, D, k, alpha, a, v2, v3)[1]*int(6))
    I1 = (I1[0].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)]),I1[1].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)]))
    assert I_array[0] == I1

    v = [Vertex() for i in range(5)]
    v[0].link_vertex(v[1])
    v[1].link_vertex(v[2])
    v[2].link_vertex(v[3])
    v[0].link_vertex(v[4])
    v[4].link_vertex(v[3])
    v[1].add_outgoing()
    v[2].add_outgoing()
    v[4].add_outgoing()
    g2 = Graph(v)
    g2.label_edges(q,[p,q-p-r,r],k)
    I2 = (g2.integral(f, D, k, alpha, a, v2, v3)[0],g2.integral(f, D, k, alpha, a, v2, v3)[1]*int(6))
    I2 = (I2[0].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)]), I2[1].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)]))
    assert I_array[1] == I2

    v = [Vertex() for i in range(5)]
    v[0].link_vertex(v[1])
    v[1].link_vertex(v[2])
    v[2].link_vertex(v[3])
    v[0].link_vertex(v[4])
    v[4].link_vertex(v[3])
    v[1].add_outgoing()
    v[2].add_outgoing()
    v[4].add_outgoing()
    g3 = Graph(v)
    g3.label_edges(q,[q-p-r,p,r],k)
    I3 = (g3.integral(f, D, k, alpha, a, v2, v3)[0], g3.integral(f, D, k, alpha, a, v2, v3)[1]*int(6))
    I3 = (I3[0].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)]), I3[1].subs([(dot_qp,0), (dot_pk,0),(dot_qr,0), (dot_pr,0),(dot_rk,0),(p**2,0),(r**2,0)]))
    assert I_array[4] == I3

test_integrate2()
test_integrate3()
test_label_edges()
test_freq_integral()
test_multiplicity()
test_integral()
test_integrals_symmet()
print('All tests passed succesfully!')