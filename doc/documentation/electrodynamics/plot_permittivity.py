# Creates: Au.yml.png
import numpy as np
import matplotlib.pyplot as plt
from ase.units import _hplanck, _c, _e, Hartree

from gpaw.fdtd.polarizable_material import PermittivityPlus

_eps0_au = 1.0 / (4.0 * np.pi)


def eV_from_um(um_i):
    return _hplanck / _e * _c / (um_i * 1e-6)


def plot(fname, fiteps):
    with open(fname, 'r') as yml:
        for line in yml:
            if line.strip().startswith('data'):
                data_ij = np.array([[float(d) for d in line.split()]
                                    for line in yml])

    energy_j = eV_from_um(data_ij[:, 0])
    n_j = data_ij[:, 1]
    k_j = data_ij[:, 2]
    eps_j = (n_j ** 2 - k_j ** 2) + 1.0j * 2 * n_j * k_j

    energy_e = np.linspace(1.0, 6.0, 100)
    fiteps_e = np.array([fiteps.value(energy / Hartree) / _eps0_au
                         for energy in energy_e])

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(energy_j, eps_j.real, 'bv', label='data')
    plt.plot(energy_e, fiteps_e.real, 'b-', label='fit')
    plt.xlim(energy_e.min(), energy_e.max())
    # plt.ylim(fiteps_e.real.min(), fiteps_e.real.max())
    plt.ylim(-70, 0)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Real($\epsilon$)')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(energy_j, eps_j.imag, 'bv')
    plt.plot(energy_e, fiteps_e.imag, 'b-')
    plt.xlim(energy_e.min(), energy_e.max())
    # plt.ylim(fiteps_e.imag.min(), fiteps_e.imag.max())
    plt.ylim(0, 7)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Imaginary($\epsilon$)')
    plt.tight_layout()
    plt.savefig('%s.png' % fname)

# Permittivity of Gold
# Source:
# http://refractiveindex.info/?shelf=main&book=Au&page=Johnson
# Direct download link:
# wget http://refractiveindex.info/database/main/Au/Johnson.yml -O Au.yml
ymlfname = 'Au.yml'

# Fit to the permittivity
# J. Chem. Phys. 135, 084121 (2011); http://dx.doi.org/10.1063/1.3626549
fiteps = PermittivityPlus(data=[[0.2350, 0.1551, 95.62],
                                [0.4411, 0.1480, -12.55],
                                [0.7603, 1.946, -40.89],
                                [1.161, 1.396, 17.22],
                                [2.946, 1.183, 15.76],
                                [4.161, 1.964, 36.63],
                                [5.747, 1.958, 22.55],
                                [7.912, 1.361, 81.04]])
plot(ymlfname, fiteps)
