from ase.lattice.hexagonal import Graphene

from gpaw import GPAW, PW, FermiDirac
from gpaw.response.df import DielectricFunction
from ase.units import Hartree
from gpaw.test import equal

import os

system = Graphene(symbol='C',
                  latticeconstant={'a': 2.467710, 'c': 1.0},
                  size=(1, 1, 1))
system.pbc = (1, 1, 0)
system.center(axis=2, vacuum=4.0)

nk = 6
nkrefine = 5

kpt_refine = {'center': [1. / 3, 1. / 3, 0.],
              'size': [nkrefine, nkrefine, 1],
              'reduce_symmetry': False,
              'q': [1. / (nk * nkrefine), 0, 0]}

if not os.path.exists('graphene.gpw'):
    calc = GPAW(mode=PW(ecut=400),
                nbands=8,
                xc='PBE',
                kpts={'size': [nk, nk, 1], 'gamma': True},
                experimental={'kpt_refine': kpt_refine},
                occupations=FermiDirac(0.026))
    system.set_calculator(calc)
    system.get_potential_energy()
    calc.write('graphene.gpw', 'all')

pbc = system.pbc

df = DielectricFunction('graphene.gpw', eta=25e-3, domega0=0.01)
alpha0x_w, alphax_w = df.get_polarizability(q_c=[1 / (nk * nkrefine), 0, 0])
omega_w = df.get_frequencies()
analyticalalpha_w = 1j / (8 * omega_w[1:] / Hartree)

# Just some hardcoded test for alpha at omega=0
equal(alphax_w[0].real, 6.705, tolerance=0.02,
      msg='Polarizability at omega=0 is wrong')

if 0:
    from matplotlib import pyplot as plt
    plt.plot(omega_w, alphax_w.real, label='GPAW real part')
    plt.plot(omega_w, alphax_w.imag, '--', label='GPAW imag part')
    plt.plot(omega_w[1:], analyticalalpha_w.imag,
             '--', label='Analytical imag part')
    plt.xlabel(r'$\hbar\omega$ (eV)')
    plt.ylabel(r'Polarizability $\mathrm{\AA}$')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 100)
    plt.show()
