# Creates: C_freq.png
import pickle

import matplotlib.pyplot as plt
import numpy as np

f = plt.figure()
xomega2 = np.array([1, 5, 10, 15, 20, 25])
color = ['go-', 'bo-', 'ko-', 'mo-', 'co-']
data = np.zeros((5, 6))

for i, domega0 in enumerate([0.01, 0.02, 0.03, 0.04, 0.05]):
    for j, omega2 in enumerate([1, 5, 10, 15, 20, 25]):
        path = 'C_g0w0_domega0_%s_omega2_%s_results.pckl' % (domega0, omega2)
        results = pickle.load(open(path, 'rb'))

        data[i, j] = results['qp'][0, 0, 1] - results['qp'][0, 0, 0]

for j, k in enumerate([0.01, 0.02, 0.03, 0.04, 0.05]):
    plt.plot(xomega2, data[j, :], color[j], label='domega0 = %s' % (k))

plt.ylim([6.9, 7.9])
plt.xlabel('omega2 (eV)')
plt.ylabel('Direct band gap (eV)')
plt.title('non-selfconsistent G0W0@LDA')
plt.legend(loc='upper right')
plt.savefig('C_freq.png')
plt.show()
