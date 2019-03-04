from gpaw.solvation.sjm import SJM, SJMPower12Potential

from ase.build import fcc111
from gpaw import FermiDirac

# Import solvation modules
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction
)

# Solvent parameters
u0 = 0.180  # eV
epsinf = 78.36  # Dielectric constant of water at 298 K
gamma = 0.00114843767916  # 18.4*1e-3 * Pascal* m
T = 298.15   # K
atomic_radii = lambda atoms: [vdw_radii[n] for n in atoms.numbers]

# Structure is created
atoms = fcc111('Au', size=(1, 1, 3))
atoms.center(axis=2, vacuum=8)
atoms.translate([0, 0, -2])

# SJM parameters
potential = 3.0
ne = 1.20
dpot = 0.05

# The calculator
calc = SJM(doublelayer={'upper_limit': 19.5},
           potential=potential,
           dpot=dpot,
           ne=ne,
           gpts=(8, 8, 64),
           poissonsolver={'dipolelayer': 'xy'},
           kpts=(1, 1, 1),
           xc='PBE',
           occupations=FermiDirac(0.1),
           cavity=EffectivePotentialCavity(
               effective_potential=SJMPower12Potential(atomic_radii, u0),
               temperature=T,
               surface_calculator=GradientSurface()),
           dielectric=LinearDielectric(epsinf=epsinf),
           interactions=[SurfaceInteraction(surface_tension=gamma)])

# Run the calculation
atoms.set_calculator(calc)
atoms.get_potential_energy()
assert abs(calc.get_electrode_potential() - potential) < dpot
elpot = calc.get_electrostatic_potential().mean(0).mean(0)
assert abs(elpot[2] - elpot[10]) < 1e-3
