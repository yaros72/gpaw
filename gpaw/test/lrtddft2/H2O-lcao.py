from __future__ import print_function
import numpy as np

from ase.build import molecule

from gpaw import GPAW
from gpaw.lrtddft2 import LrTDDFT2
from gpaw.test import equal

name = 'H2O-lcao'
atoms = molecule('H2O')
atoms.center(vacuum=4)

# Ground state
calc = GPAW(h=0.4, mode='lcao', basis='dzp', txt='%s-gs.out' % name,
            poissonsolver={'name': 'fd'},
            nbands=8, xc='LDA')
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('%s.gpw' % name, mode='all')

# LrTDDFT2
calc = GPAW('%s.gpw' % name, txt='%s-lr.out' % name)
lr = LrTDDFT2(name, calc, fxc='LDA')
lr.calculate()
results = lr.get_transitions()[0:2]

if 0:
    np.set_printoptions(precision=10)
    refstr = repr(results)
    refstr = refstr.replace('array', 'np.array')
    # Format a pep-compatible output
    refstr = ' '.join(refstr.split())
    refstr = refstr.replace('[ ', '[')
    refstr = refstr.replace(', np', ',\n       np')
    refstr = refstr.replace(', ', ',\n                 ')
    print('ref = %s' % refstr)

ref = (np.array([6.0832234654,
                 8.8741672508,
                 13.5935056665,
                 14.2916074337,
                 15.9923770524,
                 16.9925552101,
                 17.6504895168,
                 17.6925534089,
                 24.0929126532,
                 25.0027483575,
                 25.6208990274,
                 26.9649914298,
                 29.5294793981,
                 29.8440800288]),
       np.array([3.6912275735e-02,
                 1.6097104081e-23,
                 3.0995162928e-01,
                 2.1338465380e-02,
                 1.4257298381e-22,
                 6.3876242247e-02,
                 1.0288210520e-01,
                 1.7216431909e-01,
                 2.8906903842e-02,
                 3.9353952344e-01,
                 2.1927514221e-02,
                 6.7747041559e-01,
                 7.9560508308e-02,
                 1.0626657179e-02]))

tol = 1e-8
for r0, r1 in zip(results, ref):
    equal(r0, r1, tol)
