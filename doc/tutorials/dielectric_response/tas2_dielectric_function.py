# Creates: tas2_eps.png
from __future__ import division
from ase import Atoms
from ase.lattice.hexagonal import Hexagonal
import matplotlib.pyplot as plt

from gpaw import GPAW, PW, FermiDirac
from gpaw.response.df import DielectricFunction
from gpaw.mpi import world
from gpaw.bztools import find_high_symmetry_monkhorst_pack

# 1) Ground-state.

a = 3.314
c = 12.1

cell = Hexagonal(symbol='Ta', latticeconstant={'a': a, 'c': c}).get_cell()
atoms = Atoms(symbols='TaS2TaS2', cell=cell, pbc=(1, 1, 1),
              scaled_positions=[(0, 0, 1 / 4),
                                (2 / 3, 1 / 3, 1 / 8),
                                (2 / 3, 1 / 3, 3 / 8),
                                (0, 0, 3 / 4),
                                (-2 / 3, -1 / 3, 5 / 8),
                                (-2 / 3, -1 / 3, 7 / 8)])

calc = GPAW(mode=PW(600),
            xc='PBE',
            occupations=FermiDirac(width=0.01),
            kpts={'density': 5})

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('TaS2-gs.gpw')

# 2) Unoccupied bands

kpts = find_high_symmetry_monkhorst_pack('TaS2-gs.gpw', density=5.0)

responseGS = GPAW('TaS2-gs.gpw',
                  fixdensity=True,
                  kpts=kpts,
                  parallel={'band': 1},
                  nbands=60,
                  convergence={'bands': 50})

responseGS.get_potential_energy()
responseGS.write('TaS2-gsresponse.gpw', 'all')

# 3) Dielectric function

df = DielectricFunction('TaS2-gsresponse.gpw', eta=25e-3, domega0=0.01,
                        integrationmode='tetrahedron integration')

df1tetra_w, df2tetra_w = df.get_dielectric_function(direction='x')

df = DielectricFunction('TaS2-gsresponse.gpw', eta=25e-3,
                        domega0=0.01)
df1_w, df2_w = df.get_dielectric_function(direction='x')
omega_w = df.get_frequencies()

if world.rank == 0:
    plt.figure(figsize=(6, 6))
    plt.plot(omega_w, df2tetra_w.real, label='tetra Re')
    plt.plot(omega_w, df2tetra_w.imag, label='tetra Im')
    plt.plot(omega_w, df2_w.real, label='Re')
    plt.plot(omega_w, df2_w.imag, label='Im')
    plt.xlabel('Frequency (eV)')
    plt.ylabel('$\\varepsilon$')
    plt.xlim(0, 10)
    plt.ylim(-20, 20)
    plt.legend()
    plt.tight_layout()
    plt.savefig('tas2_eps.png', dpi=600)
#    plt.show()
