from gpaw.tddft import TDDFT
from gpaw.tddft.ehrenfest import EhrenfestVelocityVerlet
from gpaw.tddft.laser import CWField
from ase.units import Hartree, Bohr, AUT
from ase.io import Trajectory
from ase.parallel import parprint

name = 'h2_diss'

# Ehrenfest simulation parameters
timestep = 10.0  # timestep given in attoseconds
ndiv = 10  # write trajectory every 10 timesteps
niter = 500  # run for 500 timesteps

# TDDFT calculator with an external potential emulating an intense
# harmonic laser field aligned (CWField uses by default the z axis)
# along the H2 molecular axis.
tdcalc = TDDFT(name + '_gs.gpw',
               txt=name + '_td.txt',
               propagator='EFSICN',
               solver='BiCGStab',
               td_potential=CWField(1000 * Hartree, 1 * AUT, 10))

# For Ehrenfest dynamics, we use this object for the Velocity Verlet dynamics.
ehrenfest = EhrenfestVelocityVerlet(tdcalc)

# Trajectory to save the dynamics.
traj = Trajectory(name + '_td.traj', 'w', tdcalc.get_atoms())

# Propagates the dynamics for niter timesteps.
for i in range(1, niter + 1):
    ehrenfest.propagate(timestep)

    if tdcalc.atoms.get_distance(0, 1) > 2.0:
        # Stop simulation if H-H distance is greater than 2 A
        parprint('Dissociated!')
        break

    # Every ndiv timesteps, save an image in the trajectory file.
    if i % ndiv == 0:
        # Currently needed with Ehrenfest dynamics to save energy,
        # forces and velocitites.
        epot = tdcalc.get_td_energy() * Hartree
        F_av = ehrenfest.F * Hartree / Bohr  # forces
        v_av = ehrenfest.v * Bohr / AUT  # velocities
        atoms = tdcalc.atoms.copy()
        # Needed to save the velocities to the trajectory:
        atoms.set_velocities(v_av)

        traj.write(atoms, energy=epot, forces=F_av)

traj.close()
