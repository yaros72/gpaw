# Creates: h2o-bader.png
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from ase.io.cube import read_cube_data

if os.path.isfile('h2o.pckl'):
    with open('h2o.pckl', 'rb') as fd:
        dens, bader, atoms = pickle.load(fd)
else:
    dens, atoms = read_cube_data('density.cube')
    bader, atoms = read_cube_data('AtIndex.cube')
    x = len(dens) // 2
    dens = dens[x]
    bader = bader[x]
    with open('h2o.pckl', 'wb') as fd:
        pickle.dump((dens, bader, atoms), fd)

x0, y0, z0 = atoms.positions[0]
y = np.linspace(0, atoms.cell[1, 1], len(dens), endpoint=False) - y0
z = np.linspace(0, atoms.cell[2, 2], len(dens[0]), endpoint=False) - z0
print(y.shape, z.shape, dens.shape, bader.shape)
print(atoms.positions)
print(dens.min(), dens.mean(), dens.max())
plt.figure(figsize=(5, 5))
plt.contourf(z, y, dens, np.linspace(0.01, 0.9, 15))
plt.contour(z, y, bader, [1.5], colors='k')
plt.axis(xmin=-2, xmax=2, ymin=-2, ymax=2)
plt.savefig('h2o-bader.png')
