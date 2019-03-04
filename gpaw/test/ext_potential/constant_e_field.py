"""A proton in an electric field."""
from ase import Atoms
from ase.units import Hartree, Bohr

from gpaw import GPAW
from gpaw.external import ConstantElectricField

h = Atoms('H')
h.center(vacuum=2.5)
h.calc = GPAW(external=ConstantElectricField(1.0),  # 1 V / Ang
              charge=1,
              txt='h.txt')
e = h.get_potential_energy()
f1 = h.get_forces()[0, 2]
h[0].z += 0.001
de = h.get_potential_energy() - e
f2 = -de / 0.001
print(f1, f2)
assert abs(f1 - 1) < 1e-4
assert abs(f2 - 1) < 5e-3

# Check writing and reading:
h.calc.write('h')
vext = GPAW('h', txt=None).hamiltonian.vext
assert abs(vext.field_v[2] - 1.0 * Bohr / Hartree) < 1e-13
