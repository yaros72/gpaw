# Creates: phases.png
import numpy as np
import pylab as plt

a = np.load('phases_7x200.npz')
phit_km = a['phi_km']
St_km = a['S_km']
Nk = len(phit_km)

phi_km = np.zeros((len(phit_km) + 1, len(phit_km[0])), float)
phi_km[1:] = phit_km
phi_km[0] = phit_km[-1]
S_km = np.zeros((len(phit_km) + 1, len(phit_km[0])), float)
S_km[1:] = St_km
S_km[0] = St_km[-1]
# S_km += 1
S_km /= 2

Nm = len(phi_km[0])
phi_km = np.tile(phi_km, (1, 2))
phi_km[:, Nm:] += 2 * np.pi
S_km = np.tile(S_km, (1, 2))

plt.figure()
plt.scatter(np.tile(np.arange(len(phi_km)), len(phi_km.T)),
            phi_km.T.reshape(-1),
            cmap=plt.get_cmap('viridis'),
            c=S_km.T.reshape(-1),
            s=5,
            marker='o')

cbar = plt.colorbar()
cbar.set_label(r'$\langle S_z\rangle/\hbar$', size=20)

# plt.xlabel(r'$k_\mathrm{y}$', size=24)
plt.ylabel(r'$\gamma_x$', size=24)
plt.xticks([0, Nk / 2, Nk],
           [r'$-\mathrm{M}$', r'$\Gamma$', r'$\mathrm{M}$'], size=20)
plt.yticks([0, np.pi, 2 * np.pi], [r'$0$', r'$\pi$', r'$2\pi$'], size=20)
plt.axis([0, Nk, 0, 2 * np.pi])
plt.tight_layout()
plt.savefig('phases.png')
# plt.show()
