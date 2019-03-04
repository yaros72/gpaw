# Creates: field.ind_Ffe.png
# -*- coding: utf-8 -*-
from gpaw.mpi import world
assert world.size == 1, 'This script should be run in serial mode (with one process).'

import numpy as np
import matplotlib.pyplot as plt

from gpaw.inducedfield.inducedfield_base import BaseInducedField
from gpaw.tddft.units import aufrequency_to_eV


# Helper function
def do_plot(d_g, ng, box, atoms):
    # Take slice of data array
    d_yx = d_g[:, :, ng[2] // 2]
    y = np.linspace(0, box[0], ng[0] + 1)[:-1]
    dy = box[0] / (ng[0] + 1)
    y += dy * 0.5
    ylabel = u'x / Å'
    x = np.linspace(0, box[1], ng[1] + 1)[:-1]
    dx = box[1] / (ng[1] + 1)
    x += dx * 0.5
    xlabel = u'y / Å'

    # Plot
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, d_yx, 40)
    plt.colorbar()
    for atom in atoms:
        pos = atom.position
        plt.scatter(pos[1], pos[0], s=50, c='k', marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([x[0], x[-1]])
    plt.ylim([y[0], y[-1]])
    ax.set_aspect('equal')

for fname, name in zip(['field.ind'], ['Classical system']):
    # Read InducedField object
    ind = BaseInducedField(fname, readmode='all')

    # Choose array
    w = 0                                      # Frequency index
    freq = ind.omega_w[w] * aufrequency_to_eV  # Frequency
    box = np.diag(ind.atoms.get_cell())        # Calculation box
    d_g = ind.Ffe_wg[w]                        # Data array
    ng = d_g.shape                             # Size of grid
    atoms = ind.atoms                          # Atoms

    do_plot(d_g, ng, box, atoms)
    plt.title('%s\nField enhancement @ %.2f eV' % (name, freq))
    plt.savefig(fname + '_Ffe.png', bbox_inches='tight')

    # Imaginary part of density
    d_g = ind.Frho_wg[w].imag
    ng = d_g.shape
    do_plot(d_g, ng, box, atoms)
    plt.title('%s\nImaginary part of induced charge density @ %.2f eV' %
              (name, freq))
    plt.savefig(fname + '_Frho.png', bbox_inches='tight')

    # Imaginary part of potential
    d_g = ind.Fphi_wg[w].imag
    ng = d_g.shape
    do_plot(d_g, ng, box, atoms)
    plt.title('%s\nImaginary part of induced potential @ %.2f eV' %
              (name, freq))
    plt.savefig(fname + '_Fphi.png', bbox_inches='tight')
