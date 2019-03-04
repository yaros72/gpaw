import sys
from ase.parallel import world
from ase.optimize import BFGS
from ase.visualize import view
from gpaw import FermiDirac
from ase.io import write, read
from ase.units import Pascal, m
from ase.neb import NEB

# Import solvation modules
from ase.data.vdw import vdw_radii
from gpaw.solvation import (
    LinearDielectric,
    EffectivePotentialCavity,
    GradientSurface,
    SurfaceInteraction)

# SJM import
from gpaw.solvation.sjm import SJM, SJMPower12Potential

# Solvent parameters from JCP 141, 174108 (2014):
u0 = 0.180  # eV
epsinf = 78.36  # dielectric constant of water at 298 K
gamma = 18.4*1e-3 * Pascal * m  # Surface tension
T = 298.15  # Temperature
atomic_radii = lambda atoms: [vdw_radii[n] for n in atoms.numbers]

# NEB parameters
nimg = 4         # Number of neb images
potential = 4.4  # Desired potential

interpolate = True # True if it is a fresh start with interpolating IS and FS
relax_end_states = True
climb = False

if interpolate:
    initial_file = 'IS_start.traj'
    final_file = 'FS_start.traj'
    ne_IS = -0.4  # First guess of charge on IS
    ne_FS = -0.3  # First guess of charge on FS
    restart_file = None
else:
    restart_file = 'neb_start.traj'


# The calculator
def calculator():
    # Obviously this calculator should be adapted
    return SJM(poissonsolver={'dipolelayer': 'xy'},
               gpts=(48, 32, 168),
               kpts=(4, 6, 1),
               xc='PBE',
               spinpol=False,
               potential=potential,
               occupations=FermiDirac(0.1),
               maxiter=400,
               cavity=EffectivePotentialCavity(
                   effective_potential=SJMPower12Potential(
                       atomic_radii, u0, H2O_layer=True),
                   temperature=T,
                   surface_calculator=GradientSurface()),
               dielectric=LinearDielectric(epsinf=epsinf),
               interactions=[SurfaceInteraction(surface_tension=gamma)],
               )


# Setting up the images
if restart_file:
    images = read(restart_file, index='-%i:' % (nimg+2))
    try:
        # This needs a slight adaptation in ase
        ne_img = [float(image.calc.results['ne']) for image in images]
    except (AttributeError, KeyError):
        # Very bad initial guesses! Should be exchanged by actual values
        ne_img = [i/10. for i in list(range(nimg + 2))]

else:
    initial = read(initial_file)
    final = read(final_file)

    # Shifts atoms in z direction so the lowest layer is equal in all images
    initial.translate([0, 0, -initial.positions[0][2] +
                      final.positions[0][2]])

    images = [initial]
    ne_img = [ne_IS]

    for i in range(nimg):
        images.append(initial.copy())
        ne_img.append(ne_IS + (ne_FS - ne_IS) * (i+1) / float(nimg + 1))

    images.append(final)
    ne_img.append(ne_FS)

# If the endstates should be relaxed in the same run
if relax_end_states:
    if relax_end_states == 'IS':
        endstates = [0]
    elif relax_end_states == 'FS':
        endstates = [-1]
    else:
        endstates = [0, -1]

    system = ['IS', 'FS']
    for i in endstates:
        images[i].set_calculator(calculator())
        images[i].calc.set(txt=system[i]+'.txt')
        images[i].calc.ne = ne_img[i]

        qn = BFGS(images[i], logfile=system[i]+'.log',
                  trajectory=system[i]+'.traj')
        qn.run(fmax=0.03)

        write(system[i]+'_relaxed_pot_%1.2f_ne_%1.5f.traj'
              % (potential, images[i].calc.ne), images[i])
else:
    for i in [0, -1]:
        images[i].calc.ne = ne_img[i]


# Combine NEB images with their respective calculators
for i in range(1, nimg+1):
    images[i].set_calculator(calculator())
    images[i].calc.set(txt='image_%i.txt' % (i))
    images[i].calc.ne = ne_img[i]

# Run the NEB
neb = NEB(images, climb=climb)
if interpolate:
    neb.interpolate()

if world.size == 1:
    view(images)
    sys.exit()
else:
    qn = BFGS(neb, logfile='neb.log', trajectory='neb.traj')
    qn.run(fmax=0.05)


write('neb_final.traj', images)
