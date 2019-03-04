from ase import Atoms
from gpaw import GPAW, PW
from gpaw.mpi import world

if world.size == 1:
    atoms = Atoms('H', cell=(2, 2, 2), pbc=True)
    atoms.calc = GPAW(mode=PW(300, force_complex_dtype=True),
                      eigensolver='direct')
    atoms.get_potential_energy()

if world.size == 2:
    atoms = Atoms('H', cell=(2, 2, 2), pbc=True)
    atoms.calc = GPAW(mode=PW(300),
                      eigensolver='direct',
                      kpts=(3, 2, 2))
    atoms.get_potential_energy()
