from __future__ import print_function
from ase.build import molecule

from gpaw import GPAW
from gpaw.cluster import Cluster
from gpaw.analyse.overlap import Overlap
import gpaw.solvation as solv
from gpaw.lrtddft import LrTDDFT
from gpaw.poisson import PoissonSolver

"""Check whether LrTDDFT in solvation works"""

h = 0.4
box = 2
nbands = 2
txt = '-'
txt = None

H2 = Cluster(molecule('H2'))
H2.minimal_box(box, h)

c1 = GPAW(h=h, txt=None, nbands=nbands)
c1.calculate(H2)

c2 = solv.SolvationGPAW(h=h,
                        txt=None,
                        nbands=nbands + 1,
                        **solv.get_HW14_water_kwargs())
c2.calculate(H2)
for poiss in [None, PoissonSolver(nn=c2.hamiltonian.poisson.nn)]:
    lr = LrTDDFT(c2, poisson=poiss)
    print(lr)
print(Overlap(c1).pseudo(c2))



