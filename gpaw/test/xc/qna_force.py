from __future__ import print_function
from gpaw import GPAW, PW
from ase.parallel import parprint
from ase.lattice.compounds import L1_2

name = 'Cu3Au'
ecut = 300
kpts = (2, 2, 2)

QNA = {'alpha': 2.0,
       'name': 'QNA',
       'stencil': 1,
       'orbital_dependent': False,
       'parameters': {'Au': (0.125, 0.1), 'Cu': (0.0795, 0.005)},
       'setup_name': 'PBE',
       'type': 'qna-gga'}

atoms = L1_2(['Au', 'Cu'], latticeconstant=3.74)
# Displace atoms to have non-zero forces in the first place
atoms[0].position[0] += 0.1

dx_array = [-0.005, 0.000, 0.005]
E = []

for i, dx in enumerate(dx_array):
    calc = GPAW(mode=PW(ecut),
                experimental={'niter_fixdensity': 2},
                xc=QNA,
                kpts=kpts,
                parallel={'domain': 1},
                txt=name + '%.0f.txt' % ecut)

    atoms[0].position[0] += dx
    atoms.set_calculator(calc)
    E.append(atoms.get_potential_energy(force_consistent=True))
    if i == 1:
        F = atoms.get_forces()[0, 0]
    atoms[0].position[0] -= dx

F_num = -(E[-1] - E[0]) / (dx_array[-1] - dx_array[0])
F_err = F_num - F

parprint('Analytical force = ', F)
parprint('Numerical  force = ', F_num)
parprint('Difference       = ', F_err)
assert abs(F_err) < 0.01, F_err
eerr = abs(E[-1] - 270.17901094)
assert eerr < 4e-6, eerr
