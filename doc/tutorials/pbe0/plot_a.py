# Creates: si-a.png
from __future__ import division
import matplotlib.pyplot as plt
import ase.db
from ase.eos import EquationOfState


def lattice_constant(volumes, energies):
    eos = EquationOfState(volumes, energies)
    v, e, B = eos.fit()
    a = (v * 4)**(1 / 3)
    return a
    
    
con = ase.db.connect('si.db')
results = []
K = list(range(2, 9))
A = []
A0 = []
for k in K:
    rows = list(con.select(k=k))
    V = [row.volume for row in rows]
    E = [row.energy for row in rows]
    E0 = [row.epbe0 for row in rows]
    A.append(lattice_constant(V, E))
    A0.append(lattice_constant(V, E0))
    
print(K, A, A0)

plt.plot(K, A, label='PBE')
plt.plot(K, A0, label='PBE0')
plt.xlabel('number of k-points')
plt.ylabel('lattice constant [Ang]')
plt.savefig('si-a.png')
