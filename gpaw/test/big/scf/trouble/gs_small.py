import numpy as np
from ase import Atoms
from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from math import sqrt

# Part 1: Ground state calculation

NBN = 7
NGr = 7
a = 2.5
c = 3.22

GR = Atoms(symbols='C2', positions=[(0.5*a,-sqrt(3)/6*a,0.0),(0.5*a, +sqrt(3)/6*a, 0.0)],
           cell=[(0.5*a,-0.5*3**0.5*a,0),
                 (0.5*a,+0.5*3**0.5*a,0),
                 (0.0,0.0,c*2.0)])
GR.set_pbc((True,True,True))

GR2 = GR.copy()
cell = GR2.get_cell()
uv = cell[0]-cell[1]
uv = uv/np.sqrt(np.sum(uv**2.0))
dist = np.array([0.5*a,-sqrt(3)/6*a]) - np.array([0.5*a, +sqrt(3)/6*a])
dist = np.sqrt(np.sum(dist**2.0))
GR2.translate(uv*dist)

BN = Atoms(symbols='BN', positions=[(0.5*a,-sqrt(3)/6*a,0.0),(0.5*a, +sqrt(3)/6*a, 0.0)],
           cell=[(0.5*a,-0.5*3**0.5*a,0),
                 (0.5*a,+0.5*3**0.5*a,0),
                 (0.0,0.0,c*2.0)])
BN.set_pbc((True,True,True))

NB = Atoms(symbols='NB', positions=[(0.5*a,-sqrt(3)/6*a,0.0),(0.5*a, +sqrt(3)/6*a, 0.0)],
           cell=[(0.5*a,-0.5*3**0.5*a,0),
                 (0.5*a,+0.5*3**0.5*a,0),
                 (0.0,0.0,c*2.0)])
NB.set_pbc((True,True,True))

GR2.translate([0,0,c])
NB.translate([0,0,(NGr + 1.0*(1 - NGr % 2))*c] + uv*dist*(NGr % 2))
BN.translate([0,0,(NGr + 1.0*(NGr % 2))*c] + uv*dist*(NGr % 2))

GRBN = (GR*(1,1,(NGr % 2 + NGr // 2))
        + GR2*(1,1,(NGr // 2))
        + NB*(1,1,(NBN // 2 + (NBN % 2)*(NGr % 2) ))
        + BN*(1,1,(NBN // 2 + (NBN % 2)*(1 - NGr % 2) )))
BNNB = BN + NB
Graphite = GR + GR2

Graphite.set_pbc((True,True,True))
old_cell = GR.get_cell()
old_cell[2,2] = 2*c
Graphite.set_cell(old_cell)

BNNB.set_pbc((True,True,True))
old_cell = BN.get_cell()
old_cell[2,2] = 2*c
BNNB.set_cell(old_cell)
BNNB.center()

GRBN.set_pbc((True,True,True))
old_cell = BN.get_cell()
old_cell[2,2] = (NGr + NBN)*c
GRBN.set_cell(old_cell)


atoms = GRBN

from ase.dft import monkhorst_pack
calc = GPAW(h=0.18,
            mode=PW(600),
            kpts=monkhorst_pack((29, 29, 1)) + np.array([0., 0., 0.]),
            xc='PBE',
            occupations=FermiDirac(0.01),
            parallel={'band': 1},
            )
 
atoms.set_calculator(calc)               

ncpus = 16
