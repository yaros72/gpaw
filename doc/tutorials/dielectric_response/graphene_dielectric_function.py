# Creates: graphene_eps.png
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.utils import seterr

from gpaw import GPAW, FermiDirac
from gpaw.wavefunctions.pw import PW
from gpaw.response.df import DielectricFunction
from gpaw.mpi import world
from gpaw.bztools import find_high_symmetry_monkhorst_pack

a = 2.5
c = 3.22

pbc = [True, True, False]

GR = Atoms(symbols='C2', positions=[(0.5 * a, -np.sqrt(3) / 6 * a, 0.0),
                                    (0.5 * a, np.sqrt(3) / 6 * a, 0.0)],
           cell=[(0.5 * a, -0.5 * 3**0.5 * a, 0),
                 (0.5 * a, 0.5 * 3**0.5 * a, 0),
                 (0.0, 0.0, c * 2.0)], pbc=True)
old_cell = GR.get_cell()
old_cell[2, 2] = 2 * c
GR.set_cell(old_cell)
atoms = GR

calc = GPAW(h=0.18,
            mode=PW(400),
            kpts={'density': 10.0, 'gamma': True},
            occupations=FermiDirac(0.1))

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('gs.gpw')

kpts = find_high_symmetry_monkhorst_pack('gs.gpw', density=30,
                                         pbc=pbc)
responseGS = GPAW('gs.gpw',
                  fixdensity=True,
                  kpts=kpts,
                  parallel={'band': 1},
                  nbands=30,
                  occupations=FermiDirac(0.001),
                  convergence={'bands': 20})

responseGS.get_potential_energy()
responseGS.write('gsresponse.gpw', 'all')

df = DielectricFunction('gsresponse.gpw', eta=25e-3,
                        pbc=pbc, domega0=0.01,
                        integrationmode='tetrahedron integration')
df1tetra, df2tetra = df.get_dielectric_function(q_c=[0, 0, 0])

df = DielectricFunction('gsresponse.gpw',
                        domega0=0.01,
                        eta=25e-3)
df1, df2 = df.get_dielectric_function(q_c=[0, 0, 0])
omega_w = df.get_frequencies()

if world.rank == 0:
    plt.figure(figsize=(6, 6))
    plt.plot(omega_w, df2.imag * 2, label='Point sampling')
    plt.plot(omega_w, df2tetra.imag * 2, label='Tetrahedron')
    # Analytical result for graphene
    sigmainter = 1 / 4.  # The surface conductivity of graphene
    with seterr(divide='ignore', invalid='ignore'):
        dfanalytic = 1 + (4 * np.pi * 1j / (omega_w / Hartree) *
                          sigmainter / (c / Bohr))

    plt.plot(omega_w, dfanalytic.imag, label='Analytic')

    plt.xlabel('Frequency (eV)')
    plt.ylabel('$\\mathrm{Im}\\varepsilon$')
    plt.xlim(0, 6)
    plt.ylim(0, 50)
    plt.legend()
    plt.tight_layout()
    plt.savefig('graphene_eps.png', dpi=600)
