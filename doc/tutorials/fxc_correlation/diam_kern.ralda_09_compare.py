import numpy as np
from gpaw.test import equal

results = np.loadtxt('diam_kern.ralda_kernel_comparison.dat')

ralda_wave = results[0]
raldac = results[1]
JGMs = results[2]
CP_dyn = results[3]
range_RPA = results[4]
RPA = results[5]

equal(ralda_wave, -9.08, 0.01)
equal(raldac, -9.01, 0.01)
equal(JGMs, -9.07, 0.01)
equal(CP_dyn, -8.55, 0.01)
equal(range_RPA, -13.84, 0.01)
equal(RPA, -11.17, 0.01)
