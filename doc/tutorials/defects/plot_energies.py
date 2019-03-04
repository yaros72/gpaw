# Creates: energies.png
import numpy as np
import matplotlib.pyplot as plt

data = np.load('formation_energies.npz')

repeat = data['repeats']
uncorrected = data['uncorrected']
inv_repeat = 1 / repeat[1:]
line = np.polyfit(inv_repeat, uncorrected[1:].real, deg=1)
f = np.poly1d(line)
points = np.linspace(0, 1, 100)
corrected = data['corrected']
plt.plot(1 / repeat, uncorrected.real, 'o', label='No corrections')
plt.plot(1 / repeat, corrected.real, 'p', label='FNV corrected')
plt.plot(points, f(points), "--", color='C0')
plt.axhline(line[1], linestyle="dashed", color='C1')

plt.xlabel('Supercell size', fontsize=18)
plt.ylabel('Energy difference (eV)', fontsize=18)
plt.xlim(left=0.06)
plt.xticks(1 / repeat, [str(x) for x in repeat])
plt.legend(loc='lower left')
plt.savefig('energies.png', bbox_inches='tight', dpi=300)
