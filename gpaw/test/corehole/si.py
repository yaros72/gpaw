from __future__ import print_function
from ase import Atoms
from gpaw import GPAW, FermiDirac
from gpaw.xas import XAS, RecursionMethod
from gpaw.test import gen

# Generate setup for oxygen with half a core-hole:
gen('Si', name='hch1s', corehole=(1, 0, 0.5), gpernode=30)
a = 2.6
si = Atoms('Si', cell=(a, a, a), pbc=True)

import numpy as np
calc = GPAW(nbands=None,
            h=0.25,
            occupations=FermiDirac(width=0.05),
            setups={0: 'hch1s'})
def stopcalc():
    calc.scf.converged = 1
calc.attach(stopcalc, 1)
si.set_calculator(calc)
e = si.get_potential_energy()
niter = calc.get_number_of_iterations()
calc.write('si.gpw')

# restart from file
calc = GPAW('si.gpw')

import gpaw.mpi as mpi
if mpi.size == 1:
    xas = XAS(calc)
    x, y = xas.get_spectra()
else:
    x = np.linspace(0, 10, 50)

k = 2
calc.set(kpts=(k, k, k))
calc.initialize()
calc.set_positions(si)
assert calc.wfs.dtype == complex

r = RecursionMethod(calc)
r.run(40)
if mpi.size == 1:
    z = r.get_spectra(x)

if 0:
    import pylab as p
    p.plot(x, y[0])
    p.plot(x, sum(y))
    p.plot(x, z[0])
    p.show()

# 2p corehole
gen('Si', name='hch2p', corehole=(2, 1, 0.5), gpernode=30)
calc = GPAW(nbands=None,
            h=0.25,
            occupations=FermiDirac(width=0.05),
            setups={0: 'hch2p'})
si.set_calculator(calc)
def stopcalc():
    calc.scf.converged = True
calc.attach(stopcalc, 1)
e = si.get_potential_energy()
niter = calc.get_number_of_iterations()
calc.write('si_hch2p.gpw')

