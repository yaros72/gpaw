from gpaw import GPAW, PW
from ase.lattice.cubic import BodyCenteredCubic
from gpaw.test import equal

QNA = {'alpha': 2.0,
       'name': 'QNA',
       'orbital_dependent': False,
       'parameters': {'Fe': (0.1485, 0.005)},
       'setup_name': 'PBE',
       'type': 'qna-gga'}

atoms = BodyCenteredCubic(symbol='Fe',
                          latticeconstant=2.854,
                          pbc=(1, 1, 1))

atoms.set_initial_magnetic_moments([2, 2])

calc = GPAW(mode=PW(400),
            kpts=(3, 3, 3),
            experimental={'niter_fixdensity': 2},
            xc=QNA,
            parallel={'domain': 1},
            txt='qna_spinpol.txt')

atoms.set_calculator(calc)
atoms.get_potential_energy()
magmoms = atoms.get_magnetic_moments()

tol = 0.003
equal(2.243, magmoms[0], tol)
equal(2.243, magmoms[1], tol)
