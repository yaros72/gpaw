import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('range_results.dat')  # full range_RPA
plt.plot(A[:, 0], A[:, 1] / 2, '-o', label='RPA')

B = np.empty_like(A)

B[:, 0] = 1.0 * A[:, 0]
B[0, 1] = 0.0
B[1, 1] = -2.2929
B[2, 1] = -5.3285
B[3, 1] = -9.0291
B[4, 1] = -10.7821
B[5, 1] = -11.7281
plt.plot(B[:, 0], B[:, 1] / 2, '-o', label='Short range contribution')
plt.axhline(A[0, 1] / 2, ls='dashed')
plt.xlabel('$r_c$', fontsize=18)
plt.ylabel('Energy/Si atom(eV)', fontsize=18)
plt.legend(loc='upper right')
# show()
plt.savefig('Ec_rpa.png')
