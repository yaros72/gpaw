import numpy as np
from gpaw.defects import ElectrostaticCorrections
from ase.io import read

sigma = 1.0
q = +1
epsilons = [1.9, 1.15]
corrected = []
uncorrected = []
neutral_energy = []
repeats = [1, 2, 3, 4, 5]
for repeat in repeats:
    pristine = 'BN{0}{0}.pristine.gpw'.format(repeat)
    charged = 'BN{0}{0}.C_B_charged.gpw'.format(repeat)
    neutral = 'BN{0}{0}.C_B_neutral.gpw'.format(repeat)
    elc = ElectrostaticCorrections(pristine=pristine,
                                   charged=charged,
                                   q=q,
                                   sigma=sigma,
                                   dimensionality='2d')
    elc.set_epsilons(epsilons)
    corrected.append(elc.calculate_corrected_formation_energy())
    uncorrected.append(elc.calculate_uncorrected_formation_energy())
    neutral_energy.append(read(neutral).get_potential_energy() -
                          read(pristine).get_potential_energy())
    data = elc.collect_electrostatic_data()
    np.savez('electrostatic_data_BN_{0}{0}.npz'.format(repeat), **data)

np.savez('formation_energies_BN.npz',
         repeats=np.array(repeats),
         corrected=np.array(corrected),
         uncorrected=np.array(uncorrected),
         neutral_energy=np.array(neutral_energy))
