# Check that the LCAODOS works.

from gpaw import GPAW
from ase.build import molecule
#import pylab as pl

system = molecule('H2O')
system.center(vacuum=3.0)
system.pbc = 1

calc = GPAW(mode='lcao', basis='dzp', h=0.3, xc='oldLDA',
            kpts=[2, 1, 1], #parallel=dict(sl_auto=True),
            nbands=8,
            #kpts=[4, 4, 4]
            )

system.set_calculator(calc)
system.get_potential_energy()

from gpaw.utilities.dos import LCAODOS
# Use RestartLCAODOS if you just restarted from a file.
# Requires one diagonalization though!

lcaodos = LCAODOS(calc)

def printdos(eps, w):
    print('sum of weights', sum(w))
    print('energy, weight')
    for e0, w0 in zip(eps, w):
        print(e0, w0)
    print('-----')
eps0, w0 = lcaodos.get_orbital_pdos(0) # O 2s

printdos(eps0, w0)

assert w0[0] > 1.6 # s state on which we projected with two electrons.
assert (w0[1:] < 0.2).all(), w0 # There is another s with some weight ~ 0.16.

#eps1_n, w1_n = fold(eps * 27.211, w, 600, 0.1)
#pl.plot(eps1, w1)
#pl.show()

#for e0, w0 in zip(eps, w):
#    print e0, w0


#print eps.shape
#print w.shape

a1 = [0, 1, 2]
eps1, w1 = lcaodos.get_atomic_subspace_pdos(a1)
assert abs(2 * len(eps1) - sum(w1)) < 0.01

printdos(eps1, w1)
print('indices %s' % lcaodos.get_atom_indices(a1))

a2 = 0 # all BFs on the O atom
eps2, w2 = lcaodos.get_atomic_subspace_pdos(a2)
printdos(eps2, w2)
print('indices %s' % lcaodos.get_atom_indices(a2))
