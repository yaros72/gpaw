from ase.units import Hartree
from gpaw import GPAW
from gpaw.utilities.dos import fold
import pickle
import matplotlib.pyplot as plt

e_f = GPAW('top.gpw').get_fermi_level()

e_n, P_n = pickle.load(open('top.pickle', 'rb'))
for n in range(2, 7):
    e, ldos = fold(e_n[n] * Hartree, P_n[n], npts=2001, width=0.2)
    plt.plot(e - e_f, ldos, label='Band: ' + str(n))
plt.legend()
plt.axis([-15, 10, None, None])
plt.xlabel('Energy [eV]')
plt.ylabel('PDOS')
plt.show()
