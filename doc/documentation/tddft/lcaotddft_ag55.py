from ase.io import read
from gpaw import GPAW
from gpaw import PoissonSolver
from gpaw.occupations import FermiDirac
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.tddft.spectrum import photoabsorption_spectrum
from gpaw import setup_paths

# Set the path for the created basis set
setup_paths.insert(0, '.')

# Read the clusterm from the xyz file
atoms = read('ag55.xyz')
atoms.center(vacuum=5.0)

# Increase the accuracy of density for ground state
convergence = {'density': 1e-8}

# Increase the accuracy of PoissonSolver and
# apply multipole corrections for monopole and dipoles
poissonsolver = PoissonSolver(eps=1e-16, remove_moment=1 + 3)

# Calculate ground state in LCAO mode
calc = GPAW(xc='GLLBSC', basis='GLLBSC.dz', h=0.3, nbands=352, mode='lcao',
            convergence=convergence, poissonsolver=poissonsolver,
            occupations=FermiDirac(0.1),
            parallel={'sl_default': (8, 6, 32), 'band': 2})
atoms.set_calculator(calc)
# Relax the ground state
atoms.get_potential_energy()
# Save the intermediate ground state result to a file
calc.write('ag55_gs.gpw', mode='all')

# Time propagation
td_calc = LCAOTDDFT('ag55_gs.gpw',
                    parallel={'sl_default': (8, 6, 32), 'band': 2})
DipoleMomentWriter(td_calc, 'ag55.dm')
td_calc.absorption_kick([1e-5, 0.0, 0.0])
td_calc.propagate(20, 500)

photoabsorption_spectrum('ag55.dm', 'ag55.spec', width=0.2)
