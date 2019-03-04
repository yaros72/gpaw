import gpaw.solvation as solv
from ase import Atoms
import numpy as np
from gpaw.solvation.cavity import BAD_RADIUS_MESSAGE

atoms = Atoms('H')
atoms.center(vacuum=3.0)
kwargs = solv.get_HW14_water_kwargs()
rfun = lambda a: [np.nan]
kwargs['cavity'].effective_potential.atomic_radii = rfun
atoms.calc = solv.SolvationGPAW(xc='LDA', h=0.24, **kwargs)
try:
    atoms.get_potential_energy()
except ValueError as error:
    if not error.args[0] == BAD_RADIUS_MESSAGE:
        raise
else:
    raise AssertionError("Expected ValueError")
