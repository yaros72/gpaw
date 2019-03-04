from ase.units import _amu, _me, Bohr, AUT, Hartree
from gpaw.tddft import TDDFT
from gpaw.tddft.ehrenfest import EhrenfestVelocityVerlet
from ase.io import Trajectory
import numpy as np

name = 'graphene_h'
Ekin = 40e3  # kinetic energy of the ion (in eV)

# Adapted to the ion energy; here 4 as (probably too large!)
timestep = 16.0 * np.sqrt(10e3 / Ekin)
ekin_str = '_ek' + str(int(Ekin / 1000)) + 'k'
strbody = name + ekin_str
traj_file = strbody + '.traj'

# The parallelization options should match the number of cores, here 8.
p_bands = 2  # number of bands to parallelise over
dom_dc = (2, 2, 4)  # domain decomposition for parallelization
parallel = {'band': p_bands, 'domain': dom_dc}

tdcalc = TDDFT(name + '.gpw',
               propagator='EFSICN',
               solver='BiCGStab',
               txt=strbody + '_td.txt',
               parallel=parallel)

proj_idx = 50  # atomic index of the projectile
delta_stop = 5.0 / Bohr  # stop condition when ion is within 5 A of boundary.

# Setting the initial velocity according to the kinetic energy.
amu_to_aumass = _amu / _me
Mproj = tdcalc.atoms.get_masses()[proj_idx] * amu_to_aumass
Ekin *= 1 / Hartree
v = np.zeros((proj_idx + 1, 3))
v[proj_idx, 2] = -np.sqrt((2 * Ekin) / Mproj) * Bohr / AUT
tdcalc.atoms.set_velocities(v)

evv = EhrenfestVelocityVerlet(tdcalc)
traj = Trajectory(traj_file, 'w', tdcalc.get_atoms())

trajdiv = 1  # number of timesteps between trajectory images
densdiv = 10  # number of timesteps between saved electron densities
niters = 100 # total number of timesteps to propagate

for i in range(niters):
    # Stopping condition when projectile z coordinate passes threshold
    if evv.x[proj_idx, 2] < delta_stop:
        tdcalc.write(strbody + '_end.gpw', mode='all')
        break

    # Saving trajectory file every trajdiv timesteps
    if i % trajdiv == 0:
        F_av = evv.F * Hartree / Bohr  # forces converted from atomic units
        v_av = evv.v * Bohr / AUT  # velocities converted from atomic units
        epot = tdcalc.get_td_energy() * Hartree  # energy
        atoms = tdcalc.get_atoms().copy()
        atoms.set_velocities(v_av)

        traj.write(atoms, energy=epot, forces=F_av)

    # Saving electron density every densdiv timesteps
    if (i != 0 and i % densdiv == 0):
        tdcalc.write(strbody + '_step' + str(i) + '.gpw')

    evv.propagate(timestep)

traj.close()
