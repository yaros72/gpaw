from ase import Atoms
from gpaw import GPAW
from gpaw.test import equal


# Self-consistent calculation:
a = 2.5
slab = Atoms('Li', cell=(a, a, 2 * a), pbc=1)
slab.calc = GPAW(kpts=(3, 3, 1), txt='li.txt',
                 parallel=dict(kpt=1))
slab.get_potential_energy()
slab.calc.write('li.gpw')

# Gamma point:
e1 = slab.calc.get_eigenvalues(kpt=0)[0]
f1 = slab.calc.get_fermi_level()

# Fix density and continue:
kpts = [(0, 0, 0)]
slab.calc.set(fixdensity=True,
              nbands=5,
              kpts=kpts,
              symmetry='off',
              eigensolver='cg')
slab.get_potential_energy()
e2 = slab.calc.get_eigenvalues(kpt=0)[0]
f2 = slab.calc.get_fermi_level()

# Start from gpw-file:
calc = GPAW('li.gpw',
            txt='li2.txt',
            fixdensity=True,
            nbands=5,
            kpts=kpts,
            symmetry='off',
            eigensolver='cg')

calc.get_potential_energy()
e3 = calc.get_eigenvalues(kpt=0)[0]
f3 = slab.calc.get_fermi_level()

equal(f2, f1, 1e-10)
equal(f3, f1, 1e-10)
equal(e1, e2, 3e-5)
equal(e1, e3, 3e-5)
