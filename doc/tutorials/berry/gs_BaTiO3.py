from ase import Atoms
from ase.optimize import BFGS
from gpaw import GPAW, PW

a = Atoms('BaTiO3',
          cell=[3.98, 3.98, 4.07],
          pbc=True,
          scaled_positions=[[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.6],
                            [0.5, 0.5, 0.0],
                            [0.5, 0.0, 0.5],
                            [0.0, 0.5, 0.5]])

calc = GPAW(mode=PW(600),
            xc='PBE',
            kpts={'size': (6, 6, 6), 'gamma': True},
            txt='relax.txt')
a.set_calculator(calc)
opt = BFGS(a)
opt.run(fmax=0.01)

calc.set(symmetry='off')
a.get_potential_energy()

calc.write('BaTiO3.gpw', mode='all')
