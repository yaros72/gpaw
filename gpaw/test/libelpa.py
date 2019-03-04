import numpy as np
from gpaw.utilities.elpa import LibElpa
from gpaw.blacs import BlacsGrid
from gpaw.mpi import world

rng = np.random.RandomState(87878787)

if world.size == 1:
    shape = 1, 1
else:
    shape = world.size // 2, 2
bg = BlacsGrid(world, *shape)

M = 8
blocksize = 2

desc = bg.new_descriptor(M, M, blocksize, blocksize)
sdesc = desc.as_serial()

Aserial = sdesc.zeros()
if world.rank == 0:
    Aserial[:] = rng.rand(*Aserial.shape)
    Aserial += Aserial.T.copy()
A = desc.distribute_from_master(Aserial)
C1 = desc.zeros()
C2 = desc.zeros()
eps1 = np.zeros(M)
eps2 = np.zeros(M)

elpa = LibElpa(desc)
print(elpa)

desc.diagonalize_dc(A.copy(), C1, eps1),

eps = np.zeros(M)
elpa.diagonalize(A.copy(), C2, eps2)

print(eps1)
print(eps2)
err = np.abs(eps1 - eps2).max()
assert err < 1e-13, err
