# Creates: planaraverages.png
import numpy as np
import matplotlib.pyplot as plt
from ase.units import Bohr

data = np.load('electrostatic_data_222.npz')

z = data['z'] * Bohr
# dV = data['D_V']
# V_model = data['V_model']
V_diff = data['V_X'] - data['V_0']
# plt.plot(z, dV.real, '-', label=r'$\Delta V(z)$')
# plt.plot(z, V_model.real, '-', label='$V(z)$')
plt.plot(z, V_diff.real[0], '-',
         label=(r'$[V^{V_\mathrm{Ga}^{-3}}_\mathrm{el}(z) -'
                r'V^{0}_\mathrm{el}(z) ]$'))

constant = data['D_V_mean']
print(constant)
plt.axhline(constant, ls='dashed')
plt.axhline(0.0, ls='-', color='grey')
plt.xlabel(r'$z\enspace (\mathrm{\AA})$', fontsize=18)
plt.ylabel('Planar averages (eV)', fontsize=18)
plt.legend(loc='upper right')
plt.xlim((z[0], z[-1]))
plt.savefig('planaraverages.png', bbox_inches='tight', dpi=300)
