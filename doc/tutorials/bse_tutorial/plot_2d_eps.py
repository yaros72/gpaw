# Creates: 2d_eps.png
import matplotlib.pyplot as plt
import numpy as np

plt.figure()

a = np.loadtxt('2d_eps.dat')
plt.plot(a[:, 0], a[:, 1], 'o-', c='b', ms=10, lw=2, label='Bare')
plt.plot(a[:, 0], a[:, 2], 'o-', c='r', ms=8, lw=2, label='Truncated')

plt.xlabel(r'$q$', size=28)
plt.ylabel(r'$\epsilon_{2D}$', size=28)
plt.xticks([0, 10], [r'$\Gamma$', r'$K$'], size=24)
plt.yticks(size=20)
plt.tight_layout()
plt.legend(loc='upper right')
plt.axis([0, a[-1, 0], 0.0, 9.9])

# plt.show()
plt.savefig('2d_eps.png')
