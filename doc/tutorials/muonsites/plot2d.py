# Creates: pot_contour.png
from gpaw import restart
import matplotlib.pyplot as plt
import numpy as np

mnsi, calc = restart('mnsi.gpw', txt=None)
v = calc.get_electrostatic_potential()
a = mnsi.cell[0, 0]
n = v.shape[0]
x = y = np.linspace(0, a, n, endpoint=False)

f = plt.figure()
ax = f.add_subplot(111)
cax = ax.contour(x, y, v[:, :, n // 2], 100)
cbar = f.colorbar(cax)
ax.set_xlabel('x (Angstrom)')
ax.set_ylabel('y (Angstrom)')
ax.set_title('Pseudo-electrostatic Potential')
f.savefig('pot_contour.png')
