# Creates: BN_GW0.png
import numpy as np
import pickle
import matplotlib.pyplot as plt

result = pickle.load(open('BN_GW0_results.pckl','rb'))
nite = result['iqp'].shape[0]
gap = []
for i in range(nite):
    gap = np.append(gap,np.min(result['iqp'][i,0,:,3])-np.max(result['iqp'][i,0,:,2]))

plt.figure()
plt.plot(range(nite),gap,'ko-')
plt.ylabel('Band gap (eV)')
plt.xlabel('Iteration')
plt.savefig('BN_GW0.png')
plt.show()




