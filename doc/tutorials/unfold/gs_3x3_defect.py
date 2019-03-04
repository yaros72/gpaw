from ase.build import mx2
from gpaw import GPAW, FermiDirac

structure = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.127,
                size=(3, 3, 1), vacuum=7.5)
structure.pbc = (1, 1, 1)

# Create vacancy
del structure[2]

calc = GPAW(mode='lcao',
            basis='dzp',
            xc='LDA',
            kpts=(4, 4, 1),
            occupations=FermiDirac(0.01),
            txt='gs_3x3_defect.txt')

structure.set_calculator(calc)
structure.get_potential_energy()
calc.write('gs_3x3_defect.gpw', 'all')
