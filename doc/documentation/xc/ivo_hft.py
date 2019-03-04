"""Calculate the excitation energy of NaCl by an RSF using IVOs."""
from ase.build import molecule
from ase.units import Hartree
from gpaw import GPAW, setup_paths
from gpaw.mpi import world
from gpaw.occupations import FermiDirac
from gpaw.test import equal, gen
from gpaw.eigensolvers import RMMDIIS
from gpaw.cluster import Cluster
from gpaw.lrtddft import LrTDDFT

h = 0.3  # Gridspacing
e_singlet = 4.3
e_singlet_lr = 4.3

if setup_paths[0] != '.':
    setup_paths.insert(0, '.')

gen('Na', xcname='PBE', scalarrel=True, exx=True, yukawa_gamma=0.40)
gen('Cl', xcname='PBE', scalarrel=True, exx=True, yukawa_gamma=0.40)

c = {'energy': 0.005, 'eigenstates': 1e-2, 'density': 1e-2}
mol = Cluster(molecule('NaCl'))
mol.minimal_box(5.0, h=h)
calc = GPAW(txt='NaCl.txt', xc='LCY-PBE:omega=0.40:excitation=singlet',
            eigensolver=RMMDIIS(), h=h, occupations=FermiDirac(width=0.0),
            spinpol=False, convergence=c)
mol.set_calculator(calc)
mol.get_potential_energy()
(eps_homo, eps_lumo) = calc.get_homo_lumo()
e_ex = eps_lumo - eps_homo
equal(e_singlet, e_ex, 0.15)
calc.write('NaCl.gpw')

lr = LrTDDFT(calc, txt='LCY_TDDFT_NaCl.log', istart=6, jend=7, nspins=2)
lr.write('LCY_TDDFT_NaCl.ex.gz')
if world.rank == 0:
    lr2 = LrTDDFT('LCY_TDDFT_NaCl.ex.gz')
    lr2.diagonalize()
    ex_lr = lr2[1].get_energy() * Hartree
    equal(e_singlet_lr, e_singlet, 0.05)


