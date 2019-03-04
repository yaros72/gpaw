from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw import KohnShamConvergenceError
from gpaw.utilities.timelimit import TimeLimiter

# Atoms
atoms = molecule('Na2')
atoms.center(vacuum=4.0)

# Ground-state calculation that will never converge
maxiter = 100
calc = GPAW(mode='lcao', basis='sz(dzp)', setups='1',
            convergence={'density': 1e-100},
            maxiter=maxiter)
atoms.set_calculator(calc)

tl = TimeLimiter(calc, timelimit=0, output='scf.txt')
tl.reset('scf', min_updates=3)
try:
    atoms.get_potential_energy()
except KohnShamConvergenceError:
    assert calc.scf.maxiter < maxiter, 'TimeLimiter did not break the SCF loop'
else:
    raise AssertionError('SCF loop ended too early')
calc.write('gs.gpw', mode='all')

# Time-propagation calculation that will never finish
td_calc = LCAOTDDFT('gs.gpw')
tl = TimeLimiter(td_calc, timelimit=0, output='tddft.txt')
tl.reset('tddft', min_updates=3)
td_calc.propagate(10, maxiter - td_calc.niter)
assert td_calc.maxiter < maxiter, 'TimeLimiter did not break the TDDFT loop'
