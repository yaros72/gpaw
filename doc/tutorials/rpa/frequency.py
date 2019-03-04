from __future__ import print_function
from ase.parallel import paropen
from gpaw.xc.rpa import RPACorrelation
import numpy as np

dw = 0.5
frequencies = np.array([dw * i for i in range(200)])
weights = len(frequencies) * [dw]
weights[0] /= 2
weights[-1] /= 2
weights = np.array(weights)

rpa = RPACorrelation('N2.gpw',
                     txt='frequency_equidistant.txt',
                     frequencies=frequencies,
                     weights=weights)

Es = rpa.calculate(ecut=[50])
Es_w = rpa.E_w

with paropen('frequency_equidistant.dat', 'w') as fd:
    for w, E in zip(frequencies, Es_w):
        print(w, E.real, file=fd)
