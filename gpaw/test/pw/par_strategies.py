import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import world

ecut = 200
kpoints = [1, 1, 4]
atoms = Atoms('HLi', cell=[6, 6, 3.4], pbc=True,
              positions=[[3, 3, 0], [3, 3, 1.6]])

for xc in ['LDA', 'PBE']:
    def calculate(d, k):
        label = 'gpaw.{xc}.domain{d}.kpt{k}'.format(xc=xc, d=d, k=k)
        atoms.calc = GPAW(mode=PW(ecut),
                          xc=xc,
                          txt=label + '.txt',
                          parallel={'domain': d, 'kpt': k},
                          kpts={'size': kpoints},
                          occupations=FermiDirac(width=0.1))

        def stopcalc():
            atoms.calc.scf.converged = True
        atoms.calc.attach(stopcalc, 4)

        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        s = atoms.get_stress()
        atoms.calc.write(label + '.gpw', mode='all')
        GPAW(label + '.gpw', txt=None)
        return e, f, s

    for d in [1, 2, 4, 8]:
        for k in [1, 2]:
            if d * k > world.size:
                continue
            e2, f2, s2 = calculate(d, k)
            if d + k == 2:
                e1, f1, s1 = e2, f2, s2
            else:
                eerr = abs(e2 - e1)
                ferr = np.abs(f2 - f1).max()
                serr = np.abs(s2 - s1).max()
                if world.rank == 0:
                    print('errs', d, k, eerr, ferr, serr)
                assert eerr < 1e-11, 'bad {} energy: err={}'.format(xc, eerr)
                assert ferr < 3e-11, 'bad {} forces: err={}'.format(xc, ferr)
                assert serr < 1e-11, 'bad {} stress: err={}'.format(xc, serr)
