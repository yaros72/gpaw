# Creates: con_freq.png
import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('con_freq.dat').transpose()
plt.plot(A[0], A[1], 'o-')

plt.xlabel('Number of frequency points', fontsize=18)
plt.ylabel('Energy', fontsize=18)
plt.axis([None, None, -6.7, -6.3])
plt.savefig('con_freq.png')
