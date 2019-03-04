"""Test poisson solver for asymmetric charges."""
from __future__ import print_function
from gpaw.utilities.gauss import Gaussian
from gpaw.grid_descriptor import GridDescriptor
from gpaw.poisson import PoissonSolver


# Initialize classes
a = 20.0  # Size of cell
inv_width = 19  # inverse width of the gaussian
N = 48  # Number of grid points
center_of_charge = (a / 2, a / 2, 3 * a / 4)  # off center charge
Nc = (N, N, N)                # Number of grid points along each axis
gd = GridDescriptor(Nc, (a, a, a), 0)    # Grid-descriptor object
solver = PoissonSolver(nn=3, use_charge_center=True)
solver.set_grid_descriptor(gd)
gauss = Gaussian(gd, a=inv_width, center=center_of_charge)
test_poisson = Gaussian(gd, a=inv_width, center=center_of_charge)

# /-------------------------------------------------\
# | Check if Gaussian potentials are made correctly |
# \-------------------------------------------------/

# Array for storing the potential
pot = gd.zeros(dtype=float, global_array=False)
solver.load_gauss()
vg = test_poisson.get_gauss_pot(0)
# Get analytic functions
ng = gauss.get_gauss(0)
#    vg = solver.phi_gauss
# Solve potential numerically
niter = solver.solve(pot, ng, charge=1.0, zero_initial_phi=False)
# Determine residual
# residual = norm(pot - vg)
residual = gd.integrate((pot - vg)**2)**0.5

print('residual %s' % (
    residual))
assert residual < 1e-5  # Better than 5.x

# mpirun -np 2 python gauss_func.py --gpaw-parallel --gpaw-debug
