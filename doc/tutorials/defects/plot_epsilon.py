# Creates: dielectric_profile.png
import matplotlib.pyplot as plt
from ase.units import Bohr
from gpaw.defects import ElectrostaticCorrections

sigma = 1
q = +1
epsilons = [1.80453346, 1.14639513]
repeat = 2
pristine = 'BN{0}{0}.pristine.gpw'.format(repeat)
charged = 'BN{0}{0}.C_B_charged.gpw'.format(repeat)
neutral = 'BN{0}{0}.C_B_neutral.gpw'.format(repeat)
elc = ElectrostaticCorrections(pristine=pristine,
                               charged=charged,
                               q=q,
                               sigma=sigma,
                               dimensionality='2d')
elc.set_epsilons(epsilons)

plt.plot(elc.z_g * Bohr, elc.density_1d, linestyle='dashed', label='density')
plt.plot(elc.z_g * Bohr, elc.epsilons_z['in-plane'],
         label=r'$\varepsilon_{\parallel}(z)$')
plt.plot(elc.z_g * Bohr, elc.epsilons_z['out-of-plane'],
         label=r'$\varepsilon_{\perp}(z)$')

plt.xlabel(r'$z\enspace (\mathrm{\AA})$', fontsize=18)
plt.ylabel('Dielectric function')

plt.legend()
plt.savefig('dielectric_profile.png', bbox_inches='tight')
