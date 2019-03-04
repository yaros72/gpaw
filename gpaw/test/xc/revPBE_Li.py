from __future__ import print_function
from ase import Atoms
from ase.units import Hartree
from gpaw import GPAW, PoissonSolver, FermiDirac, Davidson, MixerSum
from gpaw.test import equal

a = 5.0
n = 24
li = Atoms('Li', magmoms=[1.0], cell=(a, a, a), pbc=True)

calc = GPAW(gpts=(n, n, n), nbands=1, xc=dict(name='oldPBE', stencil=1),
            poissonsolver=PoissonSolver(),
            mixer=MixerSum(0.6, 5, 10.0),
            eigensolver=Davidson(4),
            convergence=dict(eigenstates=4.5e-8),
            occupations=FermiDirac(0.0))
li.set_calculator(calc)
e = li.get_potential_energy() + calc.get_reference_energy()
niter_PBE = calc.get_number_of_iterations()
equal(e, -7.462 * Hartree, 1.4)

calc.set(xc=dict(name='revPBE', stencil=1))
erev = li.get_potential_energy() + calc.get_reference_energy()
niter_revPBE = calc.get_number_of_iterations()

equal(erev, -7.487 * Hartree, 1.3)
equal(e - erev, 0.025 * Hartree, 0.002 * Hartree)

print(e, erev)
energy_tolerance = 0.0002
niter_tolerance = 0
equal(e, -204.381098849, energy_tolerance)  # svnversion 5252
equal(erev, -205.012303379, energy_tolerance)  # svnversion 5252
