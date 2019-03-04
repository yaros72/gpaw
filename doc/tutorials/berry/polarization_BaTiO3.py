from gpaw.berryphase import get_polarization_phase
from ase.io import read
from ase.units import _e
import numpy as np

phi_c = get_polarization_phase('BaTiO3.gpw')
atoms = read('BaTiO3.gpw')
cell_v = np.diag(atoms.get_cell()) * 1.0e-10
V = np.prod(cell_v)
print('P:', (phi_c / (2 * np.pi) % 1) * cell_v * _e / V, 'C/m^2')
