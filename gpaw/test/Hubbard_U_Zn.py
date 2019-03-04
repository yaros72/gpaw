from __future__ import print_function
from ase import Atom

from gpaw import GPAW, FermiDirac, PoissonSolver
from gpaw.cluster import Cluster
from gpaw.test import equal

h =.35
box = 3.
energy_tolerance = 0.0004

s = Cluster([Atom('Zn')])
s.minimal_box(box, h=h)

E = {}
E_U = {}
for spin in [0, 1]:
    c = GPAW(h=h, spinpol=spin,
             xc='oldLDA',
             mode='lcao', basis='sz(dzp)',
             poissonsolver=PoissonSolver(eps=1e-7),
             parallel=dict(kpt=1),
             charge=1, occupations=FermiDirac(width=0.1, fixmagmom=spin)
             )
    s.set_calculator(c)
    E[spin] = s.get_potential_energy()
    c.set(setups=':d,3.0,1')
    E_U[spin] = s.get_potential_energy()

print("E=", E)
equal(E[0], E[1], energy_tolerance)
print("E_U=", E_U)
equal(E_U[0], E_U[1], energy_tolerance)
