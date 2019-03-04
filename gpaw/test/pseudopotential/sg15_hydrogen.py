# Test loading of sg15 setups as setups='sg15' and that the calculation
# agrees with PAW for the H2 eigenvalue.

from __future__ import print_function
from ase.build import molecule
from gpaw import GPAW, Davidson, Mixer
from gpaw.test.pseudopotential.H_sg15 import pp_text
from gpaw import setup_paths
setup_paths.insert(0, '.')
from gpaw.mpi import world

# We can't easily load a non-python file from the test suite.
# Therefore we load the pseudopotential from a Python file.
# But we want to test the pseudopotential search mechanism, therefore
# we immediately write it to a file:
if world.rank == 0:
    fd = open('H_ONCV_PBE-1.0.upf', 'w')
    fd.write(pp_text)
    fd.close()  # All right...
world.barrier()

system = molecule('H2')
system.center(vacuum=2.5)

def getkwargs():
    return dict(eigensolver=Davidson(4),
                mixer=Mixer(0.8, 5, 10.0),
                xc='oldPBE')

calc1 = GPAW(setups='sg15', h=0.13, **getkwargs())
system.set_calculator(calc1)
system.get_potential_energy()
eps1 = calc1.get_eigenvalues()

calc2 = GPAW(h=0.2, **getkwargs())
system.set_calculator(calc2)
system.get_potential_energy()
eps2 = calc2.get_eigenvalues()

err = eps2[0] - eps1[0]

# It is not the most accurate calculation ever, let's just make sure things
# are not completely messed up.
print('sg15 vs paw error', err)
assert abs(err) < 0.02  # 0.0055.... as of current test.
