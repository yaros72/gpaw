import numpy as np
from gpaw.defects import ElectrostaticCorrections

sigma = 2 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
q = -3
epsilon = 12.7
corrected = []
uncorrected = []
repeats = [1, 2, 3, 4]
for repeat in repeats:
    pristine = 'GaAs{0}{0}{0}.pristine.gpw'.format(repeat)
    charged = 'GaAs{0}{0}{0}.Ga_vac_charged.gpw'.format(repeat)
    elc = ElectrostaticCorrections(pristine=pristine,
                                   charged=charged,
                                   q=q,
                                   sigma=sigma,
                                   dimensionality='3d')
    elc.set_epsilons(epsilon)
    corrected.append(elc.calculate_corrected_formation_energy())
    uncorrected.append(elc.calculate_uncorrected_formation_energy())
    data = elc.collect_electrostatic_data()
    np.savez('electrostatic_data_{0}{0}{0}.npz'.format(repeat), **data)

np.savez('formation_energies.npz',
         repeats=np.array(repeats),
         corrected=np.array(corrected),
         uncorrected=np.array(uncorrected))
