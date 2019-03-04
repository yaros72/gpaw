import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac

ecut = 200
kpoints = [1, 1, 4]
atoms = Atoms('HLi', cell=[6, 6, 3.4], pbc=True,
              positions=[[3, 3, 0], [3, 3, 1.6]])

for xc in ['LDA', 'PBE']:
    def calculate(aug):
        atoms.calc = GPAW(mode=PW(ecut),
                          xc=xc,
                          txt='gpaw.{}.aug{}.txt'.format(xc, aug),
                          parallel={'augment_grids': aug},
                          kpts={'size': kpoints},
                          occupations=FermiDirac(width=0.1))

        def stopcalc():
            atoms.calc.scf.converged = True
        atoms.calc.attach(stopcalc, 4)

        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        s = atoms.get_stress()
        return e, f, s

    e1, f1, s1 = calculate(False)
    e2, f2, s2 = calculate(True)

    eerr = abs(e2 - e1)
    ferr = np.abs(f2 - f1).max()
    serr = np.abs(s2 - s1).max()
    if atoms.calc.wfs.world.rank == 0:
        print('errs', eerr, ferr, serr)
    assert eerr < 5e-12, 'bad {} energy: err={}'.format(xc, eerr)
    assert ferr < 5e-12, 'bad {} forces: err={}'.format(xc, ferr)
    assert serr < 5e-12, 'bad {} stress: err={}'.format(xc, serr)
