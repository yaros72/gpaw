"""Test the reading of wave functions as file references."""
from __future__ import division
from math import sqrt
import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw.mpi import world, rank


d = 3.0
atoms = Atoms('Na3',
              positions=[(0, 0, 0),
                         (0, 0, d),
                         (0, d * sqrt(3 / 4), d / 2)],
              magmoms=[1.0, 1.0, 1.0],
              cell=(3.5, 3.5, 4 + 2 / 3),
              pbc=True)

# Only a short, non-converged calcuation
conv = {'eigenstates': 1.24, 'energy': 2e-1, 'density': 1e-1}
calc = GPAW(h=0.30, kpts=(1, 1, 3),
            setups={'Na': '1'},
            nbands=3, convergence=conv)
atoms.set_calculator(calc)
e0 = atoms.get_potential_energy()
wf0 = calc.get_pseudo_wave_function(2, 1, 1, broadcast=True)

calc.write('tmp', 'all')

# Now read with single process
comm = world.new_communicator(np.array((rank,)))
calc = GPAW('tmp', communicator=comm)
wf1 = calc.get_pseudo_wave_function(2, 1, 1)
diff = np.abs(wf0 - wf1)
assert(np.all(diff < 1e-12))
