from ase import Atoms
from gpaw import GPAW
from gpaw.fdtd.poisson_fdtd import FDTDPoissonSolver
from gpaw.fdtd.polarizable_material import (PermittivityPlus,
                                            PolarizableMaterial,
                                            PolarizableSphere)
from gpaw.mpi import world
from gpaw.tddft import TDDFT
from gpaw.inducedfield.inducedfield_tddft import TDDFTInducedField
from gpaw.inducedfield.inducedfield_fdtd import (
    FDTDInducedField, calculate_hybrid_induced_field)
from gpaw.inducedfield.inducedfield_base import BaseInducedField
from gpaw.test import equal
import numpy as np

do_print_values = 0  # Use this for printing the reference values

# Accuracy
energy_eps = 0.0005
poisson_eps = 1e-12
density_eps = 1e-6

# Whole simulation cell (Angstroms)
large_cell = [20, 20, 30]

# Quantum subsystem
atom_center = np.array([10.0, 10.0, 20.0])
atoms = Atoms('Na2', [atom_center + [0.0, 0.0, -1.50],
                      atom_center + [0.0, 0.0, +1.50]])

# Permittivity file
if world.rank == 0:
    fo = open('ed.txt', 'w')
    fo.writelines(['1.20 0.20 25.0'])
    fo.close()
world.barrier()

# Classical subsystem
classical_material = PolarizableMaterial()
sphere_center = np.array([10.0, 10.0, 10.0])
classical_material.add_component(
    PolarizableSphere(permittivity=PermittivityPlus('ed.txt'),
                      center=sphere_center,
                      radius=5.0))

# Combined Poisson solver
poissonsolver = FDTDPoissonSolver(classical_material=classical_material,
                                  eps=poisson_eps,
                                  qm_spacing=0.40,
                                  cl_spacing=0.40 * 4,
                                  cell=large_cell,
                                  remove_moments=(1, 4),
                                  communicator=world,
                                  potential_coupler='Refiner')
poissonsolver.set_calculation_mode('iterate')

# Combined system
atoms.set_cell(large_cell)
atoms, qm_spacing, gpts = poissonsolver.cut_cell(atoms,
                                                 vacuum=2.50)

# Initialize GPAW
gs_calc = GPAW(gpts=gpts,
               eigensolver='cg',
               nbands=-1,
               poissonsolver=poissonsolver,
               convergence={'energy': energy_eps,
                            'density': density_eps})
atoms.set_calculator(gs_calc)

# Ground state
energy = atoms.get_potential_energy()

# Test ground state
equal(energy, -0.631881, energy_eps * gs_calc.get_number_of_electrons())

# Test floating point arithmetic errors
equal(gs_calc.hamiltonian.poisson.shift_indices_1, [4, 4, 10], 0)
equal(gs_calc.hamiltonian.poisson.shift_indices_2, [8, 8, 16], 0)

# Save state
gs_calc.write('gs.gpw', 'all')
classical_material = None
gs_calc = None

# Initialize TDDFT and FDTD
kick = [0.0, 0.0, 1.0e-3]
time_step = 10.0
iterations = 10

td_calc = TDDFT('gs.gpw')
td_calc.absorption_kick(kick_strength=kick)
td_calc.hamiltonian.poisson.set_kick(kick)

# Attach InducedFields to the calculation
frequencies = [2.05, 4.0]
width = 0.15
cl_ind = FDTDInducedField(paw=td_calc,
                          frequencies=frequencies,
                          width=width)
qm_ind = TDDFTInducedField(paw=td_calc,
                           frequencies=frequencies,
                           width=width)

# Propagate TDDFT and FDTD
td_calc.propagate(time_step, iterations // 2, 'dm.dat', 'td.gpw')
td_calc.write('td.gpw', 'all')
cl_ind.write('cl.ind')
qm_ind.write('qm.ind')
td_calc = None
cl_ind.paw = None
qm_ind.paw = None
cl_ind = None
qm_ind = None

# Restart
td_calc = TDDFT('td.gpw')
cl_ind = FDTDInducedField(filename='cl.ind',
                          paw=td_calc)
qm_ind = TDDFTInducedField(filename='qm.ind',
                           paw=td_calc)
td_calc.propagate(time_step, iterations // 2, 'dm.dat', 'td.gpw')
td_calc.write('td.gpw', 'all')
cl_ind.write('cl.ind')
qm_ind.write('qm.ind')

# Test
ref_cl_dipole_moment = [5.25374117e-14, 5.75811267e-14, 3.08349334e-02]
ref_qm_dipole_moment = [1.78620337e-11, -1.57782578e-11, 5.21368300e-01]

tol = 1e-4
equal(td_calc.hamiltonian.poisson.get_classical_dipole_moment(),
      ref_cl_dipole_moment, tol)
equal(td_calc.hamiltonian.poisson.get_quantum_dipole_moment(),
      ref_qm_dipole_moment, tol)

cl_ind.paw = None
qm_ind.paw = None
td_calc = None
cl_ind = None
qm_ind = None

# Calculate induced fields
td_calc = TDDFT('td.gpw')

# Classical subsystem
cl_ind = FDTDInducedField(filename='cl.ind', paw=td_calc)
cl_ind.calculate_induced_field(gridrefinement=2)
cl_ind.write('cl_field.ind', mode='all')

# Quantum subsystem
qm_ind = TDDFTInducedField(filename='qm.ind', paw=td_calc)
qm_ind.calculate_induced_field(gridrefinement=2)
qm_ind.write('qm_field.ind', mode='all')

# Total system, interpolate/extrapolate to a grid with spacing h
tot_ind = calculate_hybrid_induced_field(cl_ind, qm_ind, h=0.4)
tot_ind.write('tot_field.ind', mode='all')

tot_ind.paw = None
cl_ind.paw = None
qm_ind.paw = None
td_calc = None
cl_ind = None
qm_ind = None
tot_ind = None

# Test induced fields
if do_print_values:
    new_ref_values = []

    def equal(x, y, tol):
        new_ref_values.append(x)

ref_values = [72404.467117024149,
              0.520770766296,
              0.520770766299,
              0.830247064075,
              72416.234345610734,
              0.517294132489,
              0.517294132492,
              0.824704513888,
              2451.767847927681,
              0.088037476748,
              0.088037476316,
              0.123334033914,
              2454.462292798476,
              0.087537484422,
              0.087537483971,
              0.122592730690,
              76582.089818637178,
              0.589941751987,
              0.589941751804,
              0.869526245360,
              76592.175846021099,
              0.586223386358,
              0.586223386102,
              0.864478308364]

for fname in ['cl_field.ind', 'qm_field.ind', 'tot_field.ind']:
    ind = BaseInducedField(filename=fname,
                           readmode='field')
    # Estimate tolerance (worst case error accumulation)
    tol = (iterations * ind.fieldgd.integrate(ind.fieldgd.zeros() + 1.0) *
           max(density_eps, np.sqrt(poisson_eps)))
    if do_print_values:
        print('tol = %.12f' % tol)
    for w in range(len(frequencies)):
        val = ind.fieldgd.integrate(ind.Ffe_wg[w])
        equal(val, ref_values.pop(0), tol)
        for v in range(3):
            val = ind.fieldgd.integrate(np.abs(ind.Fef_wvg[w][v]))
            equal(val, ref_values.pop(0), tol)

if do_print_values:
    print('ref_values = [')
    for val in new_ref_values:
        print('              %20.12f,' % val)
    print('              ]')
