# Creates: energies_BN.png
import numpy as np
import matplotlib.pyplot as plt

data = np.load('formation_energies_BN.npz')

repeat = data['repeats']
uncorrected = data['uncorrected']
neutral = data['neutral_energy']
inv_repeat = 1 / repeat[1:]
corrected = data['corrected']
plt.plot(1 / repeat, uncorrected.real, 'o', label='No corrections')
plt.plot(1 / repeat, corrected.real, 'p', label='FNV corrected')
plt.plot(1 / repeat, neutral.real, 'd', label='Neutral system')
plt.axhline(np.mean(corrected[-3:]), linestyle="dashed", color='C1')
plt.axhline(np.mean(neutral), linestyle="dashed", color='C2')

plt.xlabel('Supercell size', fontsize=18)
plt.ylabel('Energy difference (eV)', fontsize=18)
plt.xlim(left=0.06)
plt.xticks(1 / repeat, [str(x) for x in repeat])
plt.legend(loc='upper left')
plt.savefig('energies_BN.png', bbox_inches='tight', dpi=300)
