# Creates: fig1.png
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6 / 2 ** 0.5))
data = np.loadtxt('ag55.spec')
plt.plot(data[:, 0], data[:, 1], 'k')
plt.title(r'Absorption spectrum of $Ag_{55}$ with GLLB-SC potential')
plt.xlabel('Energy (eV)')
plt.ylabel('Absorption (arbitrary units)')
plt.ylim(ymin=0)
plt.xlim(0, 10)
plt.savefig('fig1.png')
