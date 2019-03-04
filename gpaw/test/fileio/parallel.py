from gpaw import GPAW, ConvergenceError, restart
from gpaw.mixer import MixerSum
from ase.build import bulk

# bulk Fe with k-point, band, and domain parallelization
a = 2.87
atoms = bulk('Fe', 'bcc', a=a)
atoms.set_initial_magnetic_moments([2.2])
calc = GPAW(h=0.20,
            experimental={'niter_fixdensity': 2},
            eigensolver='rmm-diis',
            mixer=MixerSum(0.1, 3),
            nbands=6,
            kpts=(4, 4, 4),
            parallel={'band': 2, 'domain': (2, 1, 1)},
            maxiter=4)
atoms.set_calculator(calc)
try:
    atoms.get_potential_energy()
except ConvergenceError:
    pass
calc.write('tmp.gpw', mode='all')

# Continue calculation for few iterations
atoms, calc = restart('tmp.gpw',
                      eigensolver='rmm-diis',
                      mixer=MixerSum(0.1, 3),
                      parallel={'band': 2, 'domain': (1, 1, 2)},
                      maxiter=4)
try:
    atoms.get_potential_energy()
except ConvergenceError:
    pass
