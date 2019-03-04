from gpaw.solvation.sjm import SJM, SJMPower12Potential

from ase.build import fcc111
from gpaw import FermiDirac

# Import solvation modules
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    EffectivePotentialCavity,
    LinearDielectric,
    GradientSurface,
    SurfaceInteraction)

# Solvent parameters
u0 = 0.180  # eV
epsinf = 78.36  # Dielectric constant of water at 298 K
gamma = 0.00114843767916  # 18.4*1e-3 * Pascal* m
T = 298.15   # K


def atomic_radii(atoms):
    return [vdw_radii[n] for n in atoms.numbers]


# Structure is created
atoms = fcc111('Au', size=(1, 1, 3))
atoms.center(axis=2, vacuum=10)
atoms.translate([0, 0, -2])

# SJM parameters
potential = 3.4
ne = 0.1
dpot = 0.01

# The calculator
calc = SJM(doublelayer={'upper_limit': 23},
           potential=potential,
           dpot=dpot,
           ne=ne,
           verbose=True,

           gpts=(16, 16, 136),
           poissonsolver={'dipolelayer': 'xy'},
           kpts=(9, 9, 1),
           xc='PBE',
           txt='Au_pot_%1.2f.txt' % (potential),
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
