from ase import Atoms
from gpaw.tddft import photoabsorption_spectrum
from gpaw import PoissonSolver
from gpaw.lcaotddft.tddfpt import DensityCollector
from gpaw.lcaotddft import LCAOTDDFT

atoms = Atoms('Na8', positions=[[i * 3.0, 0, 0] for i in range(8)])
atoms.center(vacuum=5.0)

# Calculate all bands
td_calc = LCAOTDDFT(
    basis='dzp', setups={'Na': '1'}, xc='LDA', h=0.3, nbands=4,
    convergence={'density': 1e-7},
    poissonsolver=PoissonSolver(eps=1e-14, remove_moment=1 + 3 + 5))
atoms.set_calculator(td_calc)
atoms.get_potential_energy()
td_calc.write('Na8_gs.gpw', mode='all')

td_calc = LCAOTDDFT('Na8_gs.gpw')
td_calc.attach(DensityCollector('Na8.TdDen', td_calc))
td_calc.absorption_kick([1e-4, 0.0, 0.0])
td_calc.propagate(20, 1000, 'Na8.dm')

photoabsorption_spectrum('Na8.dm', 'Na8.spec', width=0.15)
