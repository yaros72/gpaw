# Check that atoms object mismatches are detected properly across CPUs.

from ase.build import molecule
from gpaw.mpi import world, synchronize_atoms

system = molecule('H2O')
synchronize_atoms(system, world)

if world.rank == 1:
    system.positions[1, 1] += 1e-8  # fail (above tolerance)
if world.rank == 2:
    system.cell[0, 0] += 1e-7  # fail (above tolerance)
if world.rank == 3:
    system.positions[1, 1] += 1e-10  # pass (below tolerance)
if world.rank == 4:
    system.cell[0, 1] += 1e-10 # pass (below tolerance)

expected_err_ranks = {1: [], 2: [1]}.get(world.size, [1, 2])

try:
    synchronize_atoms(system, world, tolerance=1e-9)
except ValueError as e:
    assert (expected_err_ranks == e.args[1]).all()
else:
    assert world.size == 1
