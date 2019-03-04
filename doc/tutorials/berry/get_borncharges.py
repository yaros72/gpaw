import json
import numpy as np

with open('borncharges-0.01.json') as fd:
    Z_avv = eval(json.load(fd))['Z_avv']
z_avv = np.zeros((len(Z_avv), 3, 3), float)
for a, Z_vv in enumerate(Z_avv):
    z_avv[a] = Z_vv
    print(np.round(Z_vv, 2))
