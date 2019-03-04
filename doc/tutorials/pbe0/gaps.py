# Creates: si-gaps.csv
from __future__ import print_function
import numpy as np
from ase.build import bulk
from ase.parallel import paropen
from gpaw.xc.exx import EXX
from gpaw.xc.tools import vxc
from gpaw import GPAW, PW

a = 5.43
si = bulk('Si', 'diamond', a)

fd = paropen('si-gaps.csv', 'w')

for k in range(2, 9, 2):
    name = 'Si-{0}'.format(k)
    si.calc = GPAW(kpts={'size': (k, k, k), 'gamma': True},
                   mode=PW(200),
                   xc='PBE',
                   convergence={'bands': 5},
                   txt=name + '.txt')
    si.get_potential_energy()
    si.calc.write(name + '.gpw', mode='all')
    
    # Range of eigenvalues:
    n1 = 3
    n2 = 5
    
    ibzkpts = si.calc.get_ibz_k_points()
    kpt_indices = []
    pbeeigs = []
    for kpt in [(0, 0, 0), (0.5, 0.5, 0)]:
        # Find k-point index:
        i = abs(ibzkpts - kpt).sum(1).argmin()
        kpt_indices.append(i)
        pbeeigs.append(si.calc.get_eigenvalues(i)[n1:n2])

    # DFT eigenvalues:
    pbeeigs = np.array(pbeeigs)
    
    # PBE contribution:
    dpbeeigs = vxc(si.calc, 'PBE')[0, kpt_indices, n1:n2]
        
    # Do PBE0 calculation:
    pbe0 = EXX(name + '.gpw',
               'PBE0',
               kpts=kpt_indices,
               bands=[n1, n2],
               txt=name + '.pbe0.txt')
    pbe0.calculate()
    
    dpbe0eigs = pbe0.get_eigenvalue_contributions()[0]
    pbe0eigs = pbeeigs - dpbeeigs + dpbe0eigs

    print('{0}, {1:.3f}, {2:.3f}, {3:.3f}, {4:.3f}'
          .format(k,
                  pbeeigs[0, 1] - pbeeigs[0, 0],
                  pbeeigs[1, 1] - pbeeigs[0, 0],
                  pbe0eigs[0, 1] - pbe0eigs[0, 0],
                  pbe0eigs[1, 1] - pbe0eigs[0, 0]),
          file=fd)
    fd.flush()
