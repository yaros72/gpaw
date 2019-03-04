# Creates: anisotropy.png
import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt('anisotropy.dat')
plt.plot(a[:, 0], (a[:, 1] - a[0, 1])* 1.0e6, '-o')

plt.xticks([0, np.pi / 2, np.pi], ['0', r'$\pi/2$', r'$\pi$'], size=20)
plt.yticks(size=16)
plt.xlabel(r'$\theta$', size=24)
plt.ylabel(r'$\mu eV$', size=24)
plt.axis([0, np.pi, None, None])
plt.tight_layout()
# plt.show()
plt.savefig('anisotropy.png')
