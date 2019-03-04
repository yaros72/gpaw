from ase import Atoms
from gpaw import GPAW
from gpaw.fdtd.poisson_fdtd import FDTDPoissonSolver
from gpaw.fdtd.polarizable_material import (PermittivityPlus,
                                            PolarizableMaterial,
                                            PolarizableSphere)
from gpaw.tddft import TDDFT, photoabsorption_spectrum
from gpaw.inducedfield.inducedfield_tddft import TDDFTInducedField
from gpaw.inducedfield.inducedfield_fdtd import FDTDInducedField
from gpaw.mpi import world
import numpy as np

# Nanosphere radius (Angstroms)
radius = 7.40

# Geometry
atom_center = np.array([30., 15., 15.])
sphere_center = np.array([15., 15., 15.])
simulation_cell = np.array([40., 30., 30.])

# Atoms object
atoms = Atoms('Na2', atom_center + np.array([[-1.5, 0.0, 0.0],
                                             [1.5, 0.0, 0.0]]))

# Permittivity of Gold
# J. Chem. Phys. 135, 084121 (2011); http://dx.doi.org/10.1063/1.3626549
eps_gold = PermittivityPlus(data=[[0.2350, 0.1551, 95.62],
                                  [0.4411, 0.1480, -12.55],
                                  [0.7603, 1.946, -40.89],
                                  [1.161, 1.396, 17.22],
                                  [2.946, 1.183, 15.76],
                                  [4.161, 1.964, 36.63],
                                  [5.747, 1.958, 22.55],
                                  [7.912, 1.361, 81.04]])

# 3) Nanosphere + Na2
classical_material = PolarizableMaterial()
classical_material.add_component(PolarizableSphere(center=sphere_center,
                                                   radius=radius,
                                                   permittivity=eps_gold))

# Combined Poisson solver
poissonsolver = FDTDPoissonSolver(classical_material=classical_material,
                                  qm_spacing=0.5,
                                  cl_spacing=2.0,
                                  cell=simulation_cell,
                                  communicator=world,
                                  remove_moments=(1, 1))
poissonsolver.set_calculation_mode('iterate')

# Combined system
atoms.set_cell(simulation_cell)
atoms, qm_spacing, gpts = poissonsolver.cut_cell(atoms, vacuum=4.0)

# Initialize GPAW
gs_calc = GPAW(gpts=gpts,
               nbands=-1,
               poissonsolver=poissonsolver)
atoms.set_calculator(gs_calc)

# Ground state
energy = atoms.get_potential_energy()

# Save state
gs_calc.write('gs.gpw', 'all')

# Initialize TDDFT and FDTD
kick = [0.001, 0.000, 0.000]
time_step = 10
iterations = 1500

td_calc = TDDFT('gs.gpw')
td_calc.absorption_kick(kick_strength=kick)
td_calc.hamiltonian.poisson.set_kick(kick)

# Attach InducedFields to the calculation
frequencies = [2.05, 2.60]
width = 0.15
cl_ind = FDTDInducedField(paw=td_calc,
                          frequencies=frequencies,
                          width=width)
qm_ind = TDDFTInducedField(paw=td_calc,
                           frequencies=frequencies,
                           width=width)

# Propagate TDDFT and FDTD
td_calc.propagate(time_step, iterations, 'dm.dat', 'td.gpw')

# Save results
td_calc.write('td.gpw', 'all')
cl_ind.write('cl.ind')
qm_ind.write('qm.ind')

photoabsorption_spectrum('dm.dat', 'spec.3.dat', width=width)
