import numpy as np
from gpaw.matrix import Matrix, matrix_matrix_multiply as mmm
from gpaw.mpi import world

N = 4
G = 7
# A0 = Matrix(N, N, dist=(world.new_communicator([0]), 1, 1))
A0 = Matrix(N, G, dist=(world, 1, 1), dtype=complex)
if world.rank == 0:
    A0.array[:, 4:] = 1j
    A0.array[:, :4] = np.diag(np.arange(N) + 1)
A = Matrix(N, G, dist=(world, world.size, 1), dtype=complex)
B = Matrix(N, G, dist=(world, world.size, 1), dtype=complex)
C = Matrix(N, N, dist=(world, world.size, 1), dtype=complex)
C0 = Matrix(N, N, dist=(world, 1, 1), dtype=complex)
A0.redist(A)
print(A.array)
A0.redist(B)
mmm(2.0, A, 'N', A, 'C', 0.0, C)
C.redist(C0)
print(C0.array)
C.array[:] = 777
mmm(2.0, A, 'N', A, 'C', 0.0, C, symmetric=True)
C.redist(C0)
print(C0.array)

N = 5
G = 7
A = Matrix(N, N, dist=(world, world.size, 1), dtype=complex)
B = Matrix(N, G, dist=(world, world.size, 1), dtype=complex)
C = Matrix(N, G, dist=(world, world.size, 1), dtype=complex)
A.array[:] = 1.0
B.array[:] = 1.0
mmm(1.0, A, 'N', B, 'N', 0.0, C)
