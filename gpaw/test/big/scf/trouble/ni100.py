ncpus = 4
from ase.build import fcc100
from gpaw import GPAW
from gpaw.utilities import h2gpts
atoms = fcc100(symbol='Ni', size=(1, 1, 9), a=3.52, vacuum=5.5)
atoms.set_initial_magnetic_moments([0.6] * len(atoms))
gpts = h2gpts(0.18, atoms.cell, idiv=8)
atoms.calc = GPAW(gpts=gpts,
                  kpts=(8, 8, 1),
                  xc='PBE')
