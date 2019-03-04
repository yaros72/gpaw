import pickle
import numpy as np
from gpaw.test import equal

g0w0 = 'MoS2_g0w0_80_results.pckl'
g0w0g = 'MoS2_g0w0g_40_results.pckl'

res_g0w0 = pickle.load(open(g0w0, 'rb'),encoding='bytes')
res_g0w0g = pickle.load(open(g0w0g, 'rb'),encoding='bytes')

equal(np.array([0.765, 2.248, 5.944, 5.944]), res_g0w0['qp'][0,0,3:7], 0.01)
equal(np.array([1.147, 2.640, 6.417, 6.418]), res_g0w0g['qp'][0,0,3:7], 0.01)
