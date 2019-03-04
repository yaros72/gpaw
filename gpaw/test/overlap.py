from __future__ import print_function
import numpy as np
from ase.build import molecule
from ase.parallel import parprint, world

from gpaw import GPAW
from gpaw.test import equal
from gpaw.cluster import Cluster
from gpaw.analyse.overlap import Overlap
from gpaw.lrtddft import LrTDDFT

"""Evaluate the overlap between two independent calculations

Differences are forced by different eigensolvers and differing number
of Kohn-Sham states.
""" 

h = 0.4
box = 2
nbands = 4
txt = '-'
txt = None
np.set_printoptions(precision=3, suppress=True)

H2 = Cluster(molecule('H2'))
H2.minimal_box(box, h)

c1 = GPAW(h=h, txt=txt, eigensolver='dav', nbands=nbands,
          convergence={'eigenstates':nbands})
c1.calculate(H2)
lr1 = LrTDDFT(c1)

parprint('sanity --------')
ov = Overlap(c1).pseudo(c1)
parprint('pseudo(normalized):\n', ov)
ov = Overlap(c1).pseudo(c1, False)
parprint('pseudo(not normalized):\n', ov)
ov = Overlap(c1).full(c1)
parprint('full:\n', ov)
equal(ov[0], np.eye(ov[0].shape[0], dtype=ov.dtype), 1e-10)


def show(c2):
    c2.calculate(H2)
    ov = Overlap(c1).pseudo(c2)
    parprint('wave function overlap (pseudo):\n', ov)
    ov = Overlap(c1).full(c2)
    parprint('wave function overlap (full):\n', ov)
    lr2 = LrTDDFT(c2)
    ovkss = lr1.kss.overlap(ov[0], lr2.kss)
    parprint('KSSingles overlap:\n', ovkss)
    ovlr = lr1.overlap(ov[0], lr2)
    parprint('LrTDDFT overlap:\n', ovlr)


parprint('cg --------')
c2 = GPAW(h=h, txt=txt, eigensolver='cg', nbands=nbands + 1,
          convergence={'eigenstates':nbands + 1})
show(c2)

parprint('spin --------')
H2.set_initial_magnetic_moments([1, -1])
c2 = GPAW(h=h, txt=txt, spinpol=True, nbands=nbands + 1,
          parallel={'domain': world.size},
          convergence={'eigenstates':nbands + 1})
H2.set_initial_magnetic_moments([0, 0])
try:
    show(c2)
except AssertionError:
    parprint('Not ready')
if 1:
    ov = Overlap(c1).pseudo(c2, otherspin=1)
    parprint('wave function overlap (pseudo other spin):\n', ov)
    
parprint('k-points --------')

H2.set_pbc([1, 1, 1])
c1 = GPAW(h=h, txt=txt, nbands=nbands,
          kpts=(1,1,3),
          #parallel={'domain': world.size},
          convergence={'eigenstates':nbands})
c1.calculate(H2)
c2 = GPAW(h=h, txt=txt, nbands=nbands + 1,
          kpts=(1,1,3),
          #parallel={'domain': world.size},
          convergence={'eigenstates':nbands + 1})
try:
    show(c2)
except (AssertionError, IndexError) as e:
    parprint('Not ready', e)
