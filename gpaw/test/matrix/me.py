import time
import numpy as np
from gpaw.wavefunctions.arrays import UniformGridWaveFunctions
from gpaw.grid_descriptor import GridDescriptor
from gpaw.mpi import world, serial_comm
from gpaw.matrix import Matrix

S = world.size
B = 9
gd = GridDescriptor([32, 28, 112], [5, 5, 20], comm=serial_comm)
w = UniformGridWaveFunctions(B, gd, complex, dist=(world, S, 1))
#w.matrix.array.real[:] = np.arange(world.rank * B // S + 1,
#                                   world.rank * B // S + 1 + B // S)[:, None]
w.array.imag[:] = np.random.uniform(-1, 1, w.array.shape)
w.array.real[:] = np.random.uniform(-1, 1, w.array.shape)
S0 = Matrix(B, B, complex, dist=(world, S, 1))
S0.array[:] = 42
S = Matrix(B, B, complex, dist=(world, S, 1))
S.array[:] = 42
t0 = time.time()
for i in range(1):
    w.matrix_elements(w, symmetric=True, cc=True, out=S0)
t1 = time.time() - t0
#print(S.array, world.rank, S.array.shape)
t0 = time.time()
for i in range(1):
    w.matrix_elements(symmetric=True, cc=True, out=S)
t2 = time.time() - t0
print(t1, t2)
#print(time.time() - t0)
#print(S.array, world.rank, S.array.shape)
world.barrier()
#print(S.array.real - S0.array.real, world.rank, S.array.shape)
#print(S.array.real, world.rank, S.array.shape)
world.barrier()
#print(S.array.real - S0.array.real, world.rank, S.array.shape)
print(S0.array.real, world.rank, S.array.shape)
world.barrier()
print(abs(S.array - S0.array), world.rank, S.array.shape)
