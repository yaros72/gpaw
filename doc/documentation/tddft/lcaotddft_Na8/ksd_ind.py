import numpy as np

from ase.io import write
from gpaw import GPAW
from gpaw.tddft.units import au_to_eV
from gpaw.lcaotddft.ksdecomposition import KohnShamDecomposition
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix

# Load the objects
calc = GPAW('unocc.gpw', txt=None)
calc.initialize_positions()  # Initialize in order to calculate density
ksd = KohnShamDecomposition(calc, 'ksd.ulm')
dmat = DensityMatrix(calc)
fdm = FrequencyDensityMatrix(calc, dmat, 'fdm.ulm')


def do(w):
    # Select the frequency and the density matrix
    rho_uMM = fdm.FReDrho_wuMM[w]
    freq = fdm.freq_w[w]
    print('Frequency: %.2f eV' % (freq.freq * au_to_eV))
    print('Folding: %s' % freq.folding)

    # Transform the LCAO density matrix to KS basis
    rho_up = ksd.transform(rho_uMM)

    # Induced density
    rho_g = ksd.get_density([rho_up[0].imag])

    # Save as a cube file
    write('ind_%.2f.cube' % (freq.freq * au_to_eV), calc.atoms, data=rho_g)

    # Calculate dipole moment for reference
    dm_v = ksd.density.finegd.calculate_dipole_moment(rho_g, center=True)
    absorption = 2 * freq.freq / np.pi * dm_v[0] / au_to_eV * 1e5
    print('Total absorption: %.2f eV^-1' % absorption)


do(0)
do(1)
