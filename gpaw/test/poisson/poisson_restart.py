import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.tddft import TDDFT as GRIDTDDFT
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.poisson import PoissonSolver
from gpaw.poisson_extravacuum import ExtraVacuumPoissonSolver

name = 'Na2'
poissoneps = 1e-16
gpts = np.array([16, 16, 24])
# Uncomment the following line if you want to run the test with 4 cpus
# gpts *= 2


def PS(**kwargs):
    return PoissonSolver(eps=poissoneps, **kwargs)

poissonsolver_i = []

ps = PS()
poissonsolver_i.append(ps)

ps = PS(remove_moment=4)
poissonsolver_i.append(ps)

ps = ExtraVacuumPoissonSolver(gpts * 2, PS())
poissonsolver_i.append(ps)

ps = ExtraVacuumPoissonSolver(gpts * 2, PS(), PS(), 2)
poissonsolver_i.append(ps)

ps1 = ExtraVacuumPoissonSolver(gpts, PS(), PS(), 1)
ps = ExtraVacuumPoissonSolver(gpts, ps1, PS(), 1)
poissonsolver_i.append(ps)

for poissonsolver in poissonsolver_i:
    for mode in ['fd', 'lcao']:
        atoms = molecule(name)
        atoms.center(vacuum=3.0)

        # Standard ground state calculation
        # Use loose convergence criterion for speed
        calc = GPAW(nbands=2, gpts=gpts / 2, setups={'Na': '1'}, txt=None,
                    poissonsolver=poissonsolver,
                    mode=mode,
                    convergence={'energy': 1.0,
                                 'density': 1.0,
                                 'eigenstates': 1.0})
        atoms.set_calculator(calc)
        energy = atoms.get_potential_energy()
        descr = calc.hamiltonian.poisson.get_description()
        calc.write('%s_gs.gpw' % name, mode='all')

        # Restart ground state
        calc = GPAW('%s_gs.gpw' % name, txt=None)
        ps = calc.hamiltonian.poisson
        assert descr == ps.get_description(), \
            'poisson solver has changed in GPAW / %s' % mode

        # Time-propagation TDDFT
        if mode == 'lcao':
            TDDFT = LCAOTDDFT
        else:
            TDDFT = GRIDTDDFT
        calc = TDDFT('%s_gs.gpw' % name, txt=None)
        ps = calc.hamiltonian.poisson
        assert descr == ps.get_description(), \
            'poisson solver has changed in TDDFT / %s' % mode
