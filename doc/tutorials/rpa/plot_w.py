# Creates: E_w.png
import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('frequency_equidistant.dat').transpose()
plt.plot(A[0], A[1], label='Equidistant')
B = np.loadtxt('frequency_gauss16.dat').transpose()
plt.plot(B[0], B[1], 'o', label='Gauss-Legendre 16')

plt.xlabel('Frequency [eV]', fontsize=18)
plt.ylabel('Integrand', fontsize=18)
plt.axis([0, 50, None, None])
plt.legend(loc='lower right')
plt.savefig('E_w.png')
