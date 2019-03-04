import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw.lrtddft import LrTDDFT
from gpaw.inducedfield.inducedfield_lrtddft import LrTDDFTInducedField
from gpaw.poisson import FDPoissonSolver
from gpaw.test import equal

do_print_values = False  # Use this for printing the reference values
poisson_eps = 1e-12
density_eps = 1e-6

# 0) PoissonSolver
poissonsolver = FDPoissonSolver(eps=poisson_eps)

# 1) Ground state calculation with empty states
atoms = Atoms(symbols='Na2',
              positions=[(0, 0, 0), (3.0, 0, 0)],
              pbc=False)
atoms.center(vacuum=3.0)

calc = GPAW(nbands=20, h=0.6, setups={'Na': '1'}, poissonsolver=poissonsolver,
            experimental={'niter_fixdensity': 2},
            convergence={'density': density_eps})
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('na2_gs_casida.gpw', mode='all')

# 2) Casida calculation
calc = GPAW('na2_gs_casida.gpw')
istart = 0
jend = 20
lr = LrTDDFT(calc, xc='LDA', istart=istart, jend=jend)
lr.diagonalize()
lr.write('na2_lr.dat.gz')

# Start from scratch
del lr
del calc
calc = GPAW('na2_gs_casida.gpw')
# calc.initialize_positions()
# calc.set_positions()
lr = LrTDDFT('na2_lr.dat.gz')

# 3) Calculate induced field
frequencies = [1.0, 2.08]  # Frequencies of interest in eV
folding = 'Gauss'          # Folding function
width = 0.1                # Line width for folding in eV
kickdir = 0                # Kick field direction 0, 1, 2 for x, y, z
ind = LrTDDFTInducedField(paw=calc, lr=lr, frequencies=frequencies,
                          folding=folding, width=width, kickdir=kickdir)
ind.calculate_induced_field(gridrefinement=2,
                            from_density='comp',
                            poissonsolver=poissonsolver,
                            extend_N_cd=3 * np.ones((3, 2), int),
                            deextend=True)

# Estimate tolerance (worst case error accumulation)
tol = (len(lr) ** 2 * ind.fieldgd.integrate(ind.fieldgd.zeros() + 1.0) *
       max(density_eps, np.sqrt(poisson_eps)))
# tol = 0.702253185994
if do_print_values:
    print('tol = %.12f' % tol)

# Test
val1 = ind.fieldgd.integrate(ind.Ffe_wg[0])
val2 = ind.fieldgd.integrate(np.abs(ind.Fef_wvg[0][0]))
val3 = ind.fieldgd.integrate(np.abs(ind.Fef_wvg[0][1]))
val4 = ind.fieldgd.integrate(np.abs(ind.Fef_wvg[0][2]))
val5 = ind.fieldgd.integrate(ind.Ffe_wg[1])
val6 = ind.fieldgd.integrate(np.abs(ind.Fef_wvg[1][0]))
val7 = ind.fieldgd.integrate(np.abs(ind.Fef_wvg[1][1]))
val8 = ind.fieldgd.integrate(np.abs(ind.Fef_wvg[1][2]))

equal(val1, 3175.732161495840, tol)
equal(val2, 1700.727018909886, tol)
equal(val3, 1187.130921347186, tol)
equal(val4, 1187.130921347852, tol)
equal(val5, 10957.059193705705, tol)
equal(val6, 6575.777519299762, tol)
equal(val7, 4589.111152993442, tol)
equal(val8, 4589.111152996577, tol)
