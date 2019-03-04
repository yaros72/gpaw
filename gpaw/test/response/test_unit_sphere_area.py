from itertools import product

import numpy as np

from gpaw.response.integrators import TetrahedronIntegrator
from gpaw.response.chi0 import ArrayDescriptor
from gpaw.test import equal


def unit(x_c):
    return np.array([[1.]], complex)


def unit_sphere(x_c):
    return np.array([(x_c**2).sum()**0.5], float)

# Test tetrahedron integrator
cell_cv = np.eye(3)
integrator = TetrahedronIntegrator(cell_cv)
x_g = np.linspace(-1, 1, 30)
x_gc = np.array([comb for comb in product(*([x_g] * 3))])

domain = (x_gc,)
out_wxx = np.zeros((1, 1, 1), complex)
integrator.integrate(kind='spectral function',
                     domain=domain,
                     integrand=(unit, unit_sphere),
                     x=ArrayDescriptor([-1.0]),
                     out_wxx=out_wxx)

equal(out_wxx[0, 0, 0], 4 * np.pi, 1e-2,
      msg='Integrated area of unit sphere is not 4 * pi')
