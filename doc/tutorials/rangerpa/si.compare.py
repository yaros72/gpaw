import numpy as np
from gpaw.test import equal

results = np.loadtxt('range_results.dat')

rc_05 = results[1, 1]
rc_10 = results[2, 1]
rc_20 = results[3, 1]
rc_30 = results[4, 1]
rc_40 = results[5, 1]

equal(rc_05, -12.13, 0.01)
equal(rc_10, -12.05, 0.01)
equal(rc_20, -12.23, 0.01)
equal(rc_30, -12.57, 0.01)
equal(rc_40, -12.85, 0.01)
