from __future__ import print_function

import numpy as np
from gpaw.spline import Spline
from gpaw.poisson import FDPoissonSolver, FFTPoissonSolver, PoissonSolver
from gpaw.grid_descriptor import GridDescriptor
from gpaw.lfc import LocalizedFunctionsCollection as LFC

L = 2.87 / 0.529177
def f(n, p):
    N = 2 * n
    gd = GridDescriptor((N, N, N), (L, L, L))
    a = gd.zeros()
    print(a.shape)
    #p = PoissonSolver(nn=1, relax=relax)
    p.set_grid_descriptor(gd)
    cut = N / 2.0 * 0.9
    s = Spline(l=0, rmax=cut, f_g=np.array([1, 0.5, 0.0]))
    c = LFC(gd, [[s], [s]])
    c.set_positions([(0, 0, 0), (0.5, 0.5, 0.5)])
    c.add(a)

    I0 = gd.integrate(a)
    a -= I0 / L**3

    b = gd.zeros()
    p.solve(b, a, charge=0, eps=1e-20)
    return gd.collect(b, broadcast=1)

b1 = f(8, FDPoissonSolver(nn=1, relax='J'))
b2 = f(8, FDPoissonSolver(nn=1, relax='GS'))
b3 = f(8, PoissonSolver(nn=1))
err1 = abs(b1[0,0,0]-b1[8,8,8])
err2 = abs(b2[0,0,0]-b2[8,8,8])
err3 = abs(b1[0,0,0]-b3[8,8,8])
print(err1)
print(err2)
print(err3)
assert err3 < 1e-10
assert err1 < 6e-16
assert err2 < 3e-6 # XXX Shouldn't this be better?

b4 = f(8, FFTPoissonSolver())
err4 = abs(b4[0, 0, 0] - b4[8, 8, 8])
print(err4)
assert err4 < 6e-16
