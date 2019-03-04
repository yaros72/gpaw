# Creates: Fe_bands.png
import numpy as np
import pylab as pl
from gpaw import GPAW
from gpaw.spinorbit import get_spinorbit_eigenvalues
#pl.rc('text', usetex=True)

calc = GPAW('Fe_bands.gpw', txt=None)

x = np.loadtxt('Fe_kpath.dat')
X = np.loadtxt('Fe_highsym.dat')

e_skn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                   for k in range(len(calc.get_ibz_k_points()))]
                  for s in range(2)])
e_snk = np.swapaxes(e_skn, 1, 2)
e_snk -= GPAW('Fe_gs.gpw').get_fermi_level()
for e_k in e_snk[0]:
    pl.plot(x, e_k, '--', c='0.5')
for e_k in e_snk[1]:
    pl.plot(x, e_k, '--', c='0.5')

e_nk, s_kvn = get_spinorbit_eigenvalues(calc, return_spin=True)
e_nk -= GPAW('Fe_gs.gpw').get_fermi_level()
s_nk = (s_kvn[:, 2].T + 1.0) / 2.0

pl.xticks(X, [r'$\Gamma$', '(010)   H   (001)', r'$\Gamma$'], size=20)
pl.yticks(size=20)
for i in range(len(X))[1:-1]:
    pl.plot(2 * [X[i]], [1.1*np.min(e_nk), 1.1*np.max(e_nk)],
            c='0.5', linewidth=0.5)

pl.scatter(np.tile(x, len(e_nk)), e_nk.reshape(-1),
           c=s_nk.reshape(-1),
           edgecolor=pl.get_cmap('jet')(s_nk.reshape(-1)),
           s=5,
           marker='+')
pl.plot([0, x[-1]], 2*[0.0], '-', c='0.5')

pl.ylabel(r'$\varepsilon_n(k)$ [eV]', size=24)
pl.axis([0, x[-1], -0.5, 0.5])
# pl.show()
pl.savefig('Fe_bands.png')
