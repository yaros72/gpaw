import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix
from gpaw.lcaotddft.ksdecomposition import KohnShamDecomposition
from gpaw.tddft.folding import frequencies
from gpaw.tddft.units import au_to_eV
from gpaw.tddft.spectrum import photoabsorption_spectrum
from gpaw.mpi import world

from gpaw.test import equal


def relative_equal(x, y, *args, **kwargs):
    return equal(np.zeros_like(x), (x - y) / x, *args, **kwargs)


# Atoms
atoms = molecule('NaCl')
atoms.center(vacuum=4.0)

# Ground-state calculation
calc = GPAW(nbands=6, h=0.4, setups=dict(Na='1'),
            basis='dzp', mode='lcao',
            poissonsolver=PoissonSolver(eps=1e-16),
            convergence={'density': 1e-8},
            txt='gs.out')
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')

# Time-propagation calculation
td_calc = LCAOTDDFT('gs.gpw', txt='td.out')
dmat = DensityMatrix(td_calc)
freqs = frequencies(range(0, 31, 5), 'Gauss', 0.1)
fdm = FrequencyDensityMatrix(td_calc, dmat, frequencies=freqs)
DipoleMomentWriter(td_calc, 'dm.dat')
kick_v = np.ones(3) * 1e-5
td_calc.absorption_kick(kick_v)
td_calc.propagate(20, 3)
fdm.write('fdm.ulm')

# Calculate reference spectrum
photoabsorption_spectrum('dm.dat', 'spec.dat', delta_e=5, width=0.1)
world.barrier()
ref_wv = np.loadtxt('spec.dat')[:, 1:]

# Calculate ground state with full unoccupied space
calc = GPAW('gs.gpw', nbands='nao', fixdensity=True, txt='unocc.out')
atoms = calc.get_atoms()
energy = atoms.get_potential_energy()
calc.write('unocc.gpw', mode='all')

# Construct KS electron-hole basis
ksd = KohnShamDecomposition(calc)
ksd.initialize(calc)
ksd.write('ksd.ulm')

# Load the objects
calc = GPAW('unocc.gpw', txt=None)
calc.initialize_positions()  # Initialize in order to calculate density
ksd = KohnShamDecomposition(calc, 'ksd.ulm')
dmat = DensityMatrix(calc)
fdm = FrequencyDensityMatrix(calc, dmat, 'fdm.ulm')

for w in range(1, len(fdm.freq_w)):
    rho_uMM = fdm.FReDrho_wuMM[w]
    rho_uMM = [rho_uMM[0] * kick_v[0] / np.sum(kick_v**2)]
    freq = fdm.freq_w[w]

    # Calculate dipole moment from density matrix
    rho_g = dmat.get_density([rho_uMM[0].imag])
    dm_v = dmat.density.finegd.calculate_dipole_moment(rho_g)
    spec_v = 2 * freq.freq / np.pi * dm_v / au_to_eV

    tol = 2e-3
    relative_equal(ref_wv[w], spec_v, tol)

    # KS transformation
    rho_up = ksd.transform(rho_uMM)

    # Calculate dipole moment from induced density
    rho_g = ksd.get_density([rho_up[0].imag])
    dm_v = ksd.density.finegd.calculate_dipole_moment(rho_g)
    spec_v = 2 * freq.freq / np.pi * dm_v / au_to_eV

    tol = 2e-3
    relative_equal(ref_wv[w], spec_v, tol)

    # Calculate dipole moment from matrix elements
    dmrho_vp = ksd.get_dipole_moment_contributions(rho_up)
    spec_vp = 2 * freq.freq / np.pi * dmrho_vp.imag / au_to_eV
    spec_v = np.sum(spec_vp, axis=1)

    tol = 2e-3
    relative_equal(ref_wv[w], spec_v, tol)
