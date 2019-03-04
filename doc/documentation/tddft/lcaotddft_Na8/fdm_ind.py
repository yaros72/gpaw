import numpy as np

from ase.io import write
from gpaw import GPAW
from gpaw.tddft.units import au_to_eV
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix

# Load the objects
calc = GPAW('unocc.gpw', txt=None)
calc.initialize_positions()  # Initialize in order to calculate density
dmat = DensityMatrix(calc)
fdm = FrequencyDensityMatrix(calc, dmat, 'fdm.ulm')


def do(w):
    # Select the frequency and the density matrix
    rho_MM = fdm.FReDrho_wuMM[w][0]
    freq = fdm.freq_w[w]
    frequency = freq.freq * au_to_eV
    print('Frequency: %.2f eV' % frequency)
    print('Folding: %s' % freq.folding)

    # Induced density
    rho_g = dmat.get_density([rho_MM.imag])

    # Save as a cube file
    write('ind_%.2f.cube' % frequency, calc.atoms, data=rho_g)

    # Calculate dipole moment for reference
    dm_v = dmat.density.finegd.calculate_dipole_moment(rho_g, center=True)
    absorption = 2 * freq.freq / np.pi * dm_v[0] / au_to_eV * 1e5
    print('Total absorption: %.2f eV^-1' % absorption)

do(0)
do(1)
