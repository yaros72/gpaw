import numpy as np
from gpaw.matrix import Matrix
from gpaw.mpi import world

N = 6
x = 0.01
A0 = Matrix(N, N, dist=(world, 1, 1), dtype=complex)
if world.rank == 0:
    A0.array[:] = np.diag(np.arange(N) + 1)
    A0.array+= np.random.uniform(-x, x, (N, N))
    A0.array += A0.array.conj().T
    B = Matrix(N, N, data=A0.array.copy())
    print(B.eigh(cc=True))
    print(B.array)
A = Matrix(N, N, dist=(world, 2, 2, 2), dtype=complex)
A0.redist(A)
print(A.array)
print(A.eigh(cc=True, scalapack=(world,2,2,2)))
print(world.rank, A.array)
A.redist(A0)
if world.rank == 0:
    print(abs(A0.array)-abs(B.array))
    print(A0.array / B.array)
