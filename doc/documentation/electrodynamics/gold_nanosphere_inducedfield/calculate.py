from ase import Atoms
from gpaw import GPAW
from gpaw.fdtd.poisson_fdtd import FDTDPoissonSolver
from gpaw.fdtd.polarizable_material import (PermittivityPlus,
                                            PolarizableMaterial,
                                            PolarizableSphere)
from gpaw.tddft import TDDFT, photoabsorption_spectrum
from gpaw.inducedfield.inducedfield_fdtd import FDTDInducedField
from gpaw.mpi import world
import numpy as np

# Nanosphere radius (Angstroms)
radius = 50.0

# Whole simulation cell (Angstroms)
large_cell = np.array([3 * radius, 3 * radius, 3 * radius])

# Permittivity of Gold
# J. Chem. Phys. 135, 084121 (2011); http://dx.doi.org/10.1063/1.3626549
gold = [[0.2350, 0.1551, 95.62],
        [0.4411, 0.1480, -12.55],
        [0.7603, 1.946, -40.89],
        [1.161, 1.396, 17.22],
        [2.946, 1.183, 15.76],
        [4.161, 1.964, 36.63],
        [5.747, 1.958, 22.55],
        [7.912, 1.361, 81.04]]

# Initialize classical material
classical_material = PolarizableMaterial()

# Classical nanosphere
classical_material.add_component(
    PolarizableSphere(center=0.5 * large_cell,
                      radius=radius,
                      permittivity=PermittivityPlus(data=gold)))

# Poisson solver
poissonsolver = FDTDPoissonSolver(classical_material=classical_material,
                                  cl_spacing=8.0,
                                  qm_spacing=1.0,
                                  cell=large_cell,
                                  communicator=world,
                                  remove_moments=(4, 1))
poissonsolver.set_calculation_mode('iterate')

# Dummy quantum system
atoms = Atoms('H', [0.5 * large_cell], cell=large_cell)
atoms, qm_spacing, gpts = poissonsolver.cut_cell(atoms)
del atoms[:]  # Remove atoms, quantum system is empty

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
iterations = 1000

td_calc = TDDFT('gs.gpw')
td_calc.absorption_kick(kick_strength=kick)
td_calc.hamiltonian.poisson.set_kick(kick)

# Attach InducedField to the calculation
frequencies = [2.45]
width = 0.0
ind = FDTDInducedField(paw=td_calc,
                       frequencies=frequencies,
                       width=width)
 
# Propagate TDDFT and FDTD
td_calc.propagate(time_step, iterations, 'dm0.dat', 'td.gpw')

# Save results
td_calc.write('td.gpw', 'all')
ind.write('td.ind')

# Spectrum
photoabsorption_spectrum('dm0.dat', 'spec.dat', width=width)

# Induced field
ind.calculate_induced_field(gridrefinement=2)
ind.write('field.ind', mode='all')
