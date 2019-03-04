from gpaw.matrix import Matrix
from gpaw.mpi import world

N = 6

if world.rank < 2:
    comm = world.new_communicator([0, 1])
else:
    comm = world.new_communicator([2, 3])

A0 = Matrix(N, N, dist=(comm, 2, 1))
A0.array[:] = world.rank
A = Matrix(N, N, dist=(world, 2, 2, 2))
A0.redist(A)
world.barrier()
print(A.array)
A0.array[:] = 117
A.redist(A0, 0)
print(A0.array)
