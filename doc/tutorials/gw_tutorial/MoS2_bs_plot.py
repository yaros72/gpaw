# Creates: MoS2_bs.png
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from gpaw.response.gw_bands import GWBands


# Initializing bands object
K = np.array([1 / 3, 1 / 3, 0])
G = np.array([0.0, 0.0, 0.0])
kpoints = np.array([G, K, G])

# GW
GW = GWBands(calc='MoS2_fulldiag.gpw',
             gw_file='MoS2_g0w0_80_results.pckl',
             kpoints=kpoints)

# Without spin-orbit
results = GW.get_gw_bands(SO=False, interpolate=True, vac=True)

# GWG
GWG = GWBands(calc='MoS2_fulldiag.gpw',
              gw_file='MoS2_g0w0g_40_results.pckl',
              kpoints=kpoints)

GWGresults = GWG.get_gw_bands(SO=False, interpolate=True, vac=True)

# Extracting data
x_x = results['x_k']
X = results['X']
eGW_kn = results['e_kn']
ef = results['ef']
eGWG_kn = GWGresults['e_kn']

# Plotting Bands
labels_K = [r'$\Gamma$', r'$K$', r'$\Gamma$']

f = plt.figure()
plt.plot(x_x, eGW_kn, '-r', linewidth=2)
plt.plot(x_x, eGWG_kn, '-g', linewidth=2)

plt.axhline(ef, color='k', linestyle='--')

for p in X:
    plt.axvline(p, color='k',linewidth=1.7)

leg_handles = [mpl.lines.Line2D([], [], linestyle='-', marker='', color='r'),
               mpl.lines.Line2D([], [], linestyle='-', marker='', color='g')]
leg_labels = [r'G$_0$W$_0$',r'G$_0$W$_0\Gamma$']
f.legend(leg_handles, leg_labels, bbox_to_anchor=(0.96,0.95),fontsize=20)

plt.xlim(0, x_x[-1])
plt.ylim([-8,0])
plt.xticks(X, labels_K, fontsize=18)
plt.yticks(fontsize=17)
plt.ylabel('Energy vs. vacuum (eV)', fontsize=24)
plt.tight_layout()
plt.savefig('MoS2_bs.png')
plt.show()
