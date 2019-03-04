# Creates: Bi2Se3_bands.png
import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW
from gpaw.spinorbit import get_spinorbit_eigenvalues, set_calculator

#plt.rc('text', usetex=True)

calc1 = GPAW('Bi2Se3_bands.gpw', txt=None)
calc2 = GPAW('gs_Bi2Se3.gpw', txt=None)
x = np.loadtxt('kpath.dat')
X = np.loadtxt('highsym.dat')

# No spin-orbit

ef = calc2.get_fermi_level()
e_kn = np.array([calc1.get_eigenvalues(kpt=k)
                 for k in range(len(calc1.get_ibz_k_points()))])
e_nk = e_kn.T
e_nk -= ef

for e_k in e_nk:
    plt.plot(x, e_k, '--', c='0.5')

# Spin-orbit calculation

e_nk = get_spinorbit_eigenvalues(calc2)
set_calculator(calc2, e_nk.T)
ef = calc2.get_fermi_level()
e_nk = get_spinorbit_eigenvalues(calc1, scale=1.0)
e_nk -= ef

plt.xticks(X, [r'$\Gamma$', 'Z', 'F', r'$\Gamma$', 'L'], size=24)
plt.yticks(size=20)
for i in range(len(X))[1:-1]:
    plt.plot(2 * [X[i]], [1.1 * np.min(e_nk), 1.1 * np.max(e_nk)],
             c='0.5', linewidth=0.5)
for e_k in e_nk:
    plt.plot(x, e_k, c='b')
plt.ylabel(r'$\varepsilon_n(k)$ [eV]', size=24)
plt.axis([0, x[-1], -1.7, 1.7])
plt.tight_layout()
# plt.show()
plt.savefig('Bi2Se3_bands.png')
