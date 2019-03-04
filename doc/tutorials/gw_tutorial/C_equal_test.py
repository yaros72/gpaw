import pickle
import numpy as np

from ase.parallel import paropen

from gpaw.test import equal


ecut_equal = np.array([[19.157, 18.639, 18.502],[11.814, 11.165, 10.974]])
for i, ecut in enumerate([100,200,300]):
    fil = pickle.load(paropen('C-g0w0_k8_ecut%s_results.pckl' %(ecut), 'rb'))
    equal(fil['qp'][0,0,1], ecut_equal[0,i], 0.01)
    equal(fil['qp'][0,0,0], ecut_equal[1,i], 0.01)

freq_equal = np.array([19.947, 19.874, 19.807, 19.978, 19.959, 19.958])
for j, omega2 in enumerate([1, 5, 10, 15, 20, 25]):
    fil = pickle.load(paropen('C_g0w0_domega0_0.02_omega2_%s_results.pckl' %(omega2),'rb'))
    equal(fil['qp'][0,0,1], freq_equal[j], 0.002)

