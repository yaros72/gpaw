import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from gpaw.jellium import JelliumSlab
from gpaw import GPAW, Mixer
from gpaw.test import equal

rs = 5.0 * Bohr  # Wigner-Seitz radius
h = 0.24          # grid-spacing
a = 8 * h        # lattice constant
v = 3 * a        # vacuum
L = 8 * a       # thickness
k = 6           # number of k-points (k*k*1)

ne = a**2 * L / (4 * np.pi / 3 * rs**3)

bc = JelliumSlab(ne, z1=v, z2=v + L)

surf = Atoms(pbc=(True, True, False),
             cell=(a, a, v + L + v))
surf.calc = GPAW(background_charge=bc,
                 poissonsolver={'dipolelayer': 'xy'},
                 xc='LDA_X+LDA_C_WIGNER',
                 eigensolver='dav',
                 kpts=[k, k, 1],
                 h=h,
                 maxiter=300,
                 convergence={'density': 1e-5},
                 mixer=Mixer(0.3, 7, 100),
                 nbands=int(ne / 2) + 15,
                 txt='surface.txt')
e = surf.get_potential_energy()

efermi = surf.calc.get_fermi_level()
# Get (x-y-averaged) electrostatic potential
# Must collect it from the CPUs
# https://listserv.fysik.dtu.dk/pipermail/gpaw-users/2014-January/002524.html
ham = surf.calc.hamiltonian
v = (ham.finegd.collect(ham.vHt_g,
                        broadcast=True) * Hartree).mean(0).mean(0)

# Get the work function
phi1 = v[-1] - efermi

equal(phi1, 2.7162036108, 1e-3)
# Reference value: Lang and Kohn, 1971, Theory of Metal Surfaces: Work function
# DOI 10.1103/PhysRevB.3.1215
# r_s = 5, work function = 2.73 eV

surf.calc.write('jellium.gpw')
