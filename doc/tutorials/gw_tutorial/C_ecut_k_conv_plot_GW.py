# Creates: C_GW.png
import pickle
import numpy as np
from ase.parallel import paropen
import matplotlib.pyplot as plt

plt.figure(figsize=(6.5, 4.5))

ecuts = np.array([100, 200, 300, 400])
color = ['ro-', 'bo-', 'go-', 'ko-', 'co-', 'mo-', 'yo-']
direct_gap = np.zeros(4)

for j, k in enumerate([6, 8, 10, 12]):
    for i, ecut in enumerate([100, 200, 300, 400]):
        try:
            fil = pickle.load(paropen('C-g0w0_k%s_ecut%s_results.pckl' %
                                      (k, ecut), 'rb'))
            direct_gap[i] = fil['qp'][0, 0, 1] - fil['qp'][0, 0, 0]
        except:
            direct_gap[i] = np.nan
    plt.plot(ecuts, direct_gap, color[j],
             label='(%sx%sx%s) k-points' % (k, k, k))

plt.xlabel('Cutoff energy (eV)')
plt.ylabel('Direct band gap (eV)')
plt.title('non-selfconsistent G0W0@LDA')
plt.legend(loc='lower right')
plt.savefig('C_GW.png')
plt.show()
