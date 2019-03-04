# Creates: 2d.png, 2d_I.png, line.png, dIdV.png
from ase.dft.stm import STM
from gpaw import GPAW
calc = GPAW('al111.gpw')
atoms = calc.get_atoms()
stm = STM(atoms)
z = 8.0
bias = 1.0
c = stm.get_averaged_current(bias, z)
x, y, h = stm.scan(bias, c, repeat=(3, 5))
# plot1
import matplotlib.pyplot as plt
plt.gca(aspect='equal')
plt.contourf(x, y, h, 40)
plt.colorbar()
plt.savefig('2d.png')
# plot2
plt.figure()
plt.gca(aspect='equal')
x, y, I = stm.scan2(bias, z, repeat=(3, 5))
plt.contourf(x, y, I, 40)
plt.colorbar()
plt.savefig('2d_I.png')
# plot3
plt.figure()
a = atoms.cell[0, 0]
x, y = stm.linescan(bias, c, [0, 0], [2 * a, 0])
plt.plot(x, y)
plt.savefig('line.png')
# plot4
plt.figure()
biasstart = -2.0
biasend = 2.0
biasstep = 0.05
bias, I, dIdV = stm.sts(0, 0, z, biasstart, biasend, biasstep)
plt.plot(bias, I, label='I')
plt.plot(bias, dIdV, label='dIdV')
plt.xlim(biasstart, biasend)
plt.legend()
plt.savefig('dIdV.png')
