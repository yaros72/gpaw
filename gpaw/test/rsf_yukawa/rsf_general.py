"""Check for tunability of gamma for yukawa potential."""
from ase import Atoms
from gpaw import GPAW, setup_paths
from gpaw.cluster import Cluster
from gpaw.eigensolvers import RMMDIIS
from gpaw.xc.hybrid import HybridXC
from gpaw.occupations import FermiDirac
from gpaw.test import equal, gen

if setup_paths[0] != '.':
    setup_paths.insert(0, '.')

for atom in ['Be']:
    gen(atom, xcname='PBE', scalarrel=True, exx=True,
        yukawa_gamma=0.83, gpernode=149)

h = 0.35
be = Cluster(Atoms('Be', positions=[[0, 0, 0]]))
be.minimal_box(3.0, h=h)

c = {'energy': 0.05, 'eigenstates': 0.05, 'density': 0.05}

IP = 8.79

xc = HybridXC('LCY-PBE', omega=0.83)
fname = 'Be_rsf.gpw'

calc = GPAW(txt='Be.txt', xc=xc, convergence=c,
            eigensolver=RMMDIIS(), h=h,
            occupations=FermiDirac(width=0.0), spinpol=False)
be.set_calculator(calc)
# energy = na2.get_potential_energy()
# calc.set(xc=xc)
energy_083 = be.get_potential_energy()
(eps_homo, eps_lumo) = calc.get_homo_lumo()
equal(eps_homo, -IP, 0.15)
xc2 = 'LCY-PBE'
energy_075 = calc.get_xc_difference(HybridXC(xc2))
equal(energy_083 - energy_075, 21.13, 0.2, 'wrong energy difference')
calc.write(fname)
calc2 = GPAW(fname)
func = calc2.get_xc_functional()
assert func['name'] == 'LCY-PBE', 'wrong name for functional'
assert func['omega'] == 0.83, 'wrong value for RSF omega'
