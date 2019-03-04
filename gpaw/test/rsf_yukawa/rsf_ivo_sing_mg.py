"""Test the calculation of the excitation energy of Na2 by RSF and IVOs."""
from ase import Atoms
from ase.units import Hartree
from gpaw import GPAW, setup_paths
from gpaw.mpi import world
from gpaw.occupations import FermiDirac
from gpaw.test import equal, gen
from gpaw.eigensolvers import RMMDIIS
from gpaw.cluster import Cluster
from gpaw.lrtddft import LrTDDFT

h = 0.35  # Gridspacing
e_singlet = 4.61
e_singlet_lr = 5.54

if setup_paths[0] != '.':
    setup_paths.insert(0, '.')

gen('Mg', xcname='PBE', scalarrel=True, exx=True, yukawa_gamma=0.38)

c = {'energy': 0.05, 'eigenstates': 3, 'density': 3}
na2 = Cluster(Atoms('Mg', positions=[[0,0,0]]))
na2.minimal_box(2.5, h=h)
calc = GPAW(txt='mg_ivo.txt', xc='LCY-PBE:omega=0.38:excitation=singlet',
            eigensolver=RMMDIIS(), h=h, occupations=FermiDirac(width=0.0),
            spinpol=False, convergence=c)
na2.set_calculator(calc)
na2.get_potential_energy()
(eps_homo, eps_lumo) = calc.get_homo_lumo()
e_ex = eps_lumo - eps_homo
equal(e_singlet, e_ex, 0.15)
calc.write('mg.gpw')
c2 = GPAW('mg.gpw')
assert c2.hamiltonian.xc.excitation == 'singlet'
lr = LrTDDFT(calc, txt='LCY_TDDFT_Mg.log', istart=4, jend=5, nspins=2)
lr.write('LCY_TDDFT_Mg.ex.gz')
if world.rank == 0:
    lr2 = LrTDDFT('LCY_TDDFT_Mg.ex.gz')
    lr2.diagonalize()
    ex_lr = lr2[1].get_energy() * Hartree
    equal(e_singlet_lr, ex_lr, 0.15)


