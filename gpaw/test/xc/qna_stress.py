import numpy as np
from gpaw import GPAW, PW, Mixer, Davidson
from ase.parallel import parprint
from ase.lattice.compounds import L1_2

name = 'Cu3Au'
structure = 'L1_2'
ecut = 300
kpts = (2, 2, 2)

QNA = {'alpha': 2.0,
       'name': 'QNA',
       'stencil': 1,
       'orbital_dependent': False,
       'parameters': {'Au': (0.125, 0.1), 'Cu': (0.0795, 0.005)},
       'setup_name': 'PBE',
       'type': 'qna-gga'}

atoms = L1_2(['Au', 'Cu'], latticeconstant=3.7)

calc = GPAW(mode=PW(ecut),
            eigensolver=Davidson(2),
            nbands='120%',
            mixer=Mixer(0.4, 7, 50.0),
            parallel=dict(domain=1),
            xc=QNA,
            kpts=kpts,
            txt=name + '.txt')

atoms.set_calculator(calc)

atoms.set_cell(np.dot(atoms.cell,
                      [[1.02, 0, 0.03],
                       [0, 0.99, -0.02],
                       [0.2, -0.01, 1.03]]),
               scale_atoms=True)

s_analytical = atoms.get_stress()
s_numerical = atoms.calc.calculate_numerical_stress(atoms, 1e-5)
s_err = s_numerical - s_analytical

parprint('Analytical stress:', s_analytical)
parprint('Numerical stress :', s_numerical)
parprint('Error in stress  :', s_err)
assert np.all(abs(s_err) < 5e-4)
