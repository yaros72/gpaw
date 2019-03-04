# Creates: bse_Si.png
import matplotlib.pyplot as plt
import numpy as np

plt.figure()

a = np.loadtxt('eps_rpa_Si.csv', delimiter=',')
plt.plot(a[:, 0], a[:, 4], label='RPA', lw=2)

a = np.loadtxt('eps_bse_Si.csv', delimiter=',')
plt.plot(a[:, 0], a[:, 2], label='BSE', lw=2)

plt.xlabel(r'$\hbar\omega\;[eV]$', size=24)
plt.ylabel(r'$\epsilon_2$', size=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()
plt.axis([2.0, 6.0, None, None])
plt.legend()

# plt.show()
plt.savefig('bse_Si.png')
