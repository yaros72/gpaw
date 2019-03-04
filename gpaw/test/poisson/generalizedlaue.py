from gpaw.poisson import NonPeriodicLauePoissonSolver, GeneralizedLauePoissonSolver, FDPoissonSolver, idst2, dst2
from gpaw.grid_descriptor import GridDescriptor
import numpy as np

gd = GridDescriptor((10,12,42), (4, 5, 20), pbc_c=(True,True,False))
poisson = GeneralizedLauePoissonSolver(nn=2)
poisson.set_grid_descriptor(gd)

poisson2 = FDPoissonSolver(nn=2, eps=1e-28)
poisson2.set_grid_descriptor(gd)

phi_g = gd.zeros()
phi2_g = gd.zeros()
rho_g = gd.zeros()
rho_g[4,5,6] = 1.0
rho_g[4,5,7] = -1.0

poisson.solve(phi_g, rho_g)
poisson2.solve(phi2_g, rho_g)
print("this", phi_g[4,5,:])
print("ref", phi2_g[4,5,:])
print("diff", phi_g[4,5,:]-phi2_g[4,5,:])

assert np.linalg.norm(phi_g-phi2_g) < 1e-10

gd = GridDescriptor((10,12,42), (10, 12, 42), pbc_c=(False,False,False))
poisson = NonPeriodicLauePoissonSolver(nn=1)
poisson.set_grid_descriptor(gd)
print("eigs", poisson.eigs_c[0])
poisson2 = FDPoissonSolver(nn=1, eps=1e-24)
poisson2.set_grid_descriptor(gd)

phi_g = gd.zeros()
phi2_g = gd.zeros()
rho_g = gd.zeros()
rho_g[:,:,1] = 1.0
rho_g[:,:,20] = -1.0
poisson.solve(phi_g, rho_g)
poisson2.solve(phi2_g, rho_g)
print("this", phi_g[4,5,:])
print("ref", phi2_g[4,5,:])
print("diff", phi_g[4,5,:]-phi2_g[4,5,:])

assert np.linalg.norm(phi_g-phi2_g) < 1e-10

X = idst2(dst2(rho_g))
diff = X-rho_g
print("dct diff", np.linalg.norm(diff))
assert np.linalg.norm(diff) < 1e-13
