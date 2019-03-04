from ase.build import bulk
from gpaw import GPAW, PW

a = 5.421
si = bulk('Si', 'fcc', a=a)
# or equivalently:
# b = a / 2
# from ase import Atoms
# si = Atoms('Si2', cell=[[0, b, b], [b, 0, b], [b, b, 0]], pbc=True,
#           scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]])

for x in [100, 200, 300, 400, 500, 600, 700, 800]:
    # for x in [0.24, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.1]:
    calc = GPAW(mode=PW(x),
                # h=x,
                xc='PBE',
                kpts=(4, 4, 4),
                txt='convergence_%s.txt' % x)

    si.calc = calc

    print(x, si.get_potential_energy())
