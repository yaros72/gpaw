# Creates: tcm_1.12.png, tcm_2.48.png, table_1.12.txt, table_2.48.txt
import numpy as np
from matplotlib import pyplot as plt

from gpaw import GPAW
from gpaw.tddft.units import au_to_eV
from gpaw.lcaotddft.ksdecomposition import KohnShamDecomposition
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix

# Load the objects
calc = GPAW('unocc.gpw', txt=None)
ksd = KohnShamDecomposition(calc, 'ksd.ulm')
dmat = DensityMatrix(calc)
fdm = FrequencyDensityMatrix(calc, dmat, 'fdm.ulm')


def do(w):
    # Select the frequency and the density matrix
    rho_uMM = fdm.FReDrho_wuMM[w]
    freq = fdm.freq_w[w]
    frequency = freq.freq * au_to_eV
    print('Frequency: %.2f eV' % frequency)
    print('Folding: %s' % freq.folding)

    # Transform the LCAO density matrix to KS basis
    rho_up = ksd.transform(rho_uMM)

    # Photoabsorption decomposition
    dmrho_vp = ksd.get_dipole_moment_contributions(rho_up)
    weight_p = 2 * freq.freq / np.pi * dmrho_vp[0].imag / au_to_eV * 1e5
    print('Total absorption: %.2f eV^-1' % np.sum(weight_p))

    # Print contributions as a table
    table = ksd.get_contributions_table(weight_p, minweight=0.1)
    print(table)
    with open('table_%.2f.txt' % frequency, 'w') as f:
        f.write('Frequency: %.2f eV\n' % frequency)
        f.write('Folding: %s\n' % freq.folding)
        f.write('Total absorption: %.2f eV^-1\n' % np.sum(weight_p))
        f.write(table)

    # Plot the decomposition as a TCM
    r = ksd.plot_TCM(weight_p,
                     occ_energy_min=-3, occ_energy_max=0.1,
                     unocc_energy_min=-0.1, unocc_energy_max=3,
                     delta_energy=0.01, sigma=0.1
                     )
    (ax_tcm, ax_occ_dos, ax_unocc_dos, ax_spec) = r

    # Plot diagonal line at the analysis frequency
    x = np.array([-4, 1])
    y = x + freq.freq * au_to_eV
    ax_tcm.plot(x, y, c='k')

    ax_occ_dos.set_title('Photoabsorption TCM of Na8 at %.2f eV' % frequency)

    # Save the plot
    plt.savefig('tcm_%.2f.png' % frequency)

do(0)
do(1)
