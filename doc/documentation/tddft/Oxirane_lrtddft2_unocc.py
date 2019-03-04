from gpaw import GPAW
from ase.io import read

L = 14  # ang
atoms = read('r-methyl-oxirane.xyz')
atoms.set_cell((L, L, L))
atoms.center()

calc = GPAW(h=0.25,
            nbands=30,
            convergence={'bands': 20},
            xc='LDA',
            maxiter=300,
            txt='unocc.txt')
atoms.set_calculator(calc)
atoms.get_potential_energy()

calc.write('r-methyl-oxirane.gpw', mode='all')
