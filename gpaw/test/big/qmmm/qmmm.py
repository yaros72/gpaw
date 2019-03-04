from math import cos, sin, pi

from ase import Atoms
from ase.calculators.tip4p import TIP4P, epsilon0, sigma0, rOH, angleHOH
from ase.calculators.qmmm import EIQMMM, LJInteractions, Embedding
from ase.constraints import FixBondLengths
from ase.optimize import LBFGS
from ase.optimize.precon import PreconLBFGS, Exp
from gpaw import GPAW

r = rOH
a = angleHOH / 180 * pi

interaction = LJInteractions({('O', 'O'): (epsilon0, sigma0)})

for selection in [[0, 1, 2], [3, 4, 5]]:
    name = ''.join(str(i) for i in selection)
    dimer = Atoms('OH2OH2',
                  [(0, 0, 0),
                   (-r * cos(a / 2), r * sin(a / 2), 0),
                   (-r * cos(a / 2), -r * sin(a / 2), 0),
                   (0, 0, 0),
                   (-r * cos(a), 0, r * sin(a)),
                   (-r, 0, 0)])
    dimer.positions[3:, 0] += 2.8
    dimer.constraints = FixBondLengths(
        [((selection[i] + 3) % 6, (selection[i - 1] + 3) % 6)
         for i in range(3)])

    dimer.calc = EIQMMM(selection,
                        GPAW(txt=name + '.txt', h=0.16),
                        TIP4P(),
                        interaction,
                        vacuum=4,
                        embedding=Embedding(rc=0.2, rc2=20, width=1),
                        output=name + '.out')
    opt = LBFGS(dimer, trajectory=name + '.traj')
    opt.run(0.02)

    monomer = dimer[selection]
    monomer.center(vacuum=4)
    monomer.calc = GPAW(txt=name + 'M.txt', h=0.16)
    opt = PreconLBFGS(monomer, precon=Exp(A=3), trajectory=name + 'M.traj')
    opt.run(0.02)
    e0 = monomer.get_potential_energy()
    be = dimer.get_potential_energy() - e0
    d = dimer.get_distance(0, 3)
    print(name, be, d)
    if name == '012':
        assert abs(be - -0.288) < 0.002
        assert abs(d - 2.76) < 0.02
    else:
        assert abs(be - -0.316) < 0.002
        assert abs(d - 2.67) < 0.02
