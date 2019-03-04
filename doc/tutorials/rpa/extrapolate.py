# Creates: extrapolate.png
from ase.utils.extrapolate import extrapolate
import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt('rpa_N2.dat')
ext, A, B, sigma = extrapolate(a[:,0], a[:,1], reg=3, plot=False)
plt.plot(a[:, 0]**(-1.5), a[:, 1], 'o', label='Calculated points')
es = np.array([e for e in a[:, 0]] + [10000])
plt.plot(es**(-1.5), A + B * es**(-1.5), '--', label='Linear regression')

t = [int(a[i, 0]) for i in range(len(a))]
plt.xticks(a[:, 0]**(-1.5), t, fontsize=12)
plt.axis([0., 150**(-1.5), None, -4.])
plt.xlabel('Cutoff energy [eV]', fontsize=18)
plt.ylabel('RPA correlation energy [eV]', fontsize=18)
plt.legend(loc='lower right')
#show()
plt.savefig('extrapolate.png')
