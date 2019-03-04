import numpy as np
from ase import Atoms
from gpaw.fdtd.poisson_fdtd import QSFDTD
from gpaw.fdtd.polarizable_material import PermittivityPlus, PolarizableMaterial, PolarizableSphere
from gpaw.mpi import world
from gpaw.test import equal

# This test does the same calculation as ed.py, but using QSFDTD wrapper instead

# Accuracy
energy_eps = 0.0005

# Whole simulation cell (Angstroms)
cell = [20, 20, 30]

# Quantum subsystem
atom_center = np.array([10.0, 10.0, 20.0])
atoms = Atoms('Na2', [atom_center + [0.0, 0.0, -1.50],
                      atom_center + [0.0, 0.0, +1.50]])

# Classical subsystem
sphere_center = np.array([10.0, 10.0, 10.0])
classical_material = PolarizableMaterial()
classical_material.add_component(PolarizableSphere(permittivity = PermittivityPlus(data = [[1.20, 0.20, 25.0]]),
                                                   center = sphere_center,
                                                   radius = 5.0
                                                   ))

# Wrap calculators
qsfdtd = QSFDTD(classical_material = classical_material,
                atoms              = atoms,
                cells              = (cell, 2.50),
                spacings           = [1.60, 0.40],
                remove_moments     = (1, 4),
                communicator       = world)

# Run
qsfdtd.ground_state('gs.gpw', eigensolver='cg', nbands=-1, convergence={'energy': energy_eps}, experimental={'niter_fixdensity': 2})
equal(qsfdtd.energy, -0.631881, energy_eps * qsfdtd.gs_calc.get_number_of_electrons())
qsfdtd.time_propagation('gs.gpw', kick_strength=[0.000, 0.000, 0.001], time_step=10, iterations=5, dipole_moment_file='dm.dat', restart_file='td.gpw')
qsfdtd.time_propagation('td.gpw', kick_strength=None, time_step=10, iterations=5, dipole_moment_file='dm.dat')

# Test
ref_cl_dipole_moment = [  5.25374117e-14,  5.75811267e-14,  3.08349334e-02]
ref_qm_dipole_moment = [  1.78620337e-11, -1.57782578e-11,  5.21368300e-01]
#print("ref_cl_dipole_moment = %s" % qsfdtd.td_calc.hamiltonian.poisson.get_classical_dipole_moment())
#print("ref_qm_dipole_moment = %s" % qsfdtd.td_calc.hamiltonian.poisson.get_quantum_dipole_moment())

tol = 1e-4
equal(qsfdtd.td_calc.hamiltonian.poisson.get_classical_dipole_moment(), ref_cl_dipole_moment, tol)
equal(qsfdtd.td_calc.hamiltonian.poisson.get_quantum_dipole_moment(), ref_qm_dipole_moment, tol)
