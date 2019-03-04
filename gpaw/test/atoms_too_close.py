from ase import Atoms
from gpaw import GPAW

atoms = Atoms('H2', [(0.0, 0.0, 0.0), 
                     (0.0, 0.0, 3.995)], 
              cell=(4, 4, 4), pbc=True)

calc = GPAW(txt=None)
atoms.set_calculator(calc)

try:
    calc.initialize(atoms)
    calc.set_positions(atoms)
except RuntimeError:
    pass
else:
    assert 2 + 2 == 5
