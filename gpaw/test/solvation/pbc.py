from gpaw.cluster import Cluster
from ase.build import molecule
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    SolvationGPAW,
    EffectivePotentialCavity,
    Power12Potential,
    LinearDielectric
)
from gpaw.solvation.poisson import ADM12PoissonSolver
import warnings

h = 0.3
vac = 3.0
u0 = .180
epsinf = 80.
T = 298.15
vdw_radii = vdw_radii.copy()
vdw_radii[1] = 1.09
atomic_radii = lambda atoms: [vdw_radii[n] for n in atoms.numbers]
convergence = {
    'energy': 0.05 / 8.,
    'density': 10.,
    'eigenstates': 10.,
}

atoms = Cluster(molecule('H2O'))
atoms.minimal_box(vac, h)
atoms.pbc = True

with warnings.catch_warnings():
    # ignore production code warning for ADM12PoissonSolver
    warnings.simplefilter("ignore")
    psolver = ADM12PoissonSolver(eps=1e-7)

atoms.calc = SolvationGPAW(
    xc='LDA', h=h, convergence=convergence,
    cavity=EffectivePotentialCavity(
        effective_potential=Power12Potential(atomic_radii=atomic_radii, u0=u0),
        temperature=T
    ),
    dielectric=LinearDielectric(epsinf=epsinf),
    poissonsolver=psolver
)
atoms.get_potential_energy()
atoms.get_forces()
