import ase.db
from ase.build import bulk
import numpy as np
from gpaw.xc.exx import EXX
from gpaw import GPAW, PW

a0 = 5.43

con = ase.db.connect('si.db')

for k in range(2, 9):
    for a in np.linspace(a0 - 0.04, a0 + 0.04, 5):
        id = con.reserve(a=a, k=k)
        if id is None:
            continue
        si = bulk('Si', 'diamond', a)
        si.calc = GPAW(kpts=(k, k, k),
                       mode=PW(400),
                       xc='PBE',
                       eigensolver='rmm-diis',
                       txt=None)
        si.get_potential_energy()
        name = 'si-{0:.2f}-{1}'.format(a, k)
        si.calc.write(name + '.gpw', mode='all')
        pbe0 = EXX(name + '.gpw', 'PBE0', txt=name + '.pbe0.txt')
        pbe0.calculate()
        epbe0 = pbe0.get_total_energy()
        
        con.write(si, a=a, k=k, epbe0=epbe0)
        del con[id]
