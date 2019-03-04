from ase import Atoms
from gpaw import GPAW, PoissonSolver
from gpaw.mixer import Mixer
from gpaw.test import equal
from gpaw.test import gen

a = 6.0
c = a / 2
#d = 1.8

elements = ['C', 'Be']
results = [0.245393619863, 9.98114719239]
electrons = [6, 3]
charges = [0,1]


for symbol in elements:
    xcname = '1.0_LDA_K_TF+1.0_LDA_X'
    g = gen(symbol, xcname=xcname, scalarrel=False, orbital_free=True,
            gpernode=75)

for element, result, e, charge in zip(elements, results, electrons, charges):
    atom = Atoms(element,
                 positions=[(c, c, c)],
                 cell=(a, a, a))

    mixer = Mixer(0.3, 5, 1)
    calc = GPAW(gpts=(32, 32, 32),
                txt='-', xc=xcname,
                poissonsolver=PoissonSolver(eps=1e-6),
                eigensolver='cg', mixer=mixer, charge=charge)

    atom.set_calculator(calc)

    E = atom.get_total_energy()
    n = calc.get_all_electron_density()

    dv = atom.get_volume() / calc.get_number_of_grid_points().prod()
    I = n.sum() * dv / 2**3

    equal(I, e, 1.0e-6)
    equal(result, E, 1.0e-3)
