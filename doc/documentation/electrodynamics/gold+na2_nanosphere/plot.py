# Creates: hybrid.png
import numpy as np
import pylab as plt

# Plot spectrum with r=0nm and r=5nm
spec0 = np.loadtxt('spec.1.dat')  # AuNP
spec1 = np.loadtxt('spec.2.dat')  # Na2
spec2 = np.loadtxt('spec.3.dat')  # AuNP+Na2

plt.figure()
plt.plot(spec0[:, 0], spec0[:, 1], 'r', label='Au nanoparticle')
plt.plot(spec1[:, 0], spec1[:, 1], 'g', label='Na$_2$')
plt.plot(spec1[:, 0], spec1[:, 1] + spec0[:, 1], 'k:',
         label='Sum of Na$_2$ and Au nanoparticle')
plt.plot(spec2[:, 0], spec2[:, 1], 'b', label='Na$_2$ near Au nanoparticle')
plt.legend(loc=1)
plt.xlabel('Energy [eV]', fontsize=12)
plt.ylabel('Dipole strength [1/eV]', fontsize=12)
plt.xlim((0, 5.0))
plt.ylim((-1, 22.5))
plt.savefig('hybrid.png')
