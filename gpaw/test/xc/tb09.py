"""Test Tran Blaha potential."""
from ase.dft.bandgap import bandgap
from ase.build import bulk
from gpaw import GPAW, PW, Mixer

def xc(name):
    return {'name': name, 'stencil': 1}

k = 8
atoms = bulk('Si')
atoms.calc = GPAW(mode=PW(300),
                  mixer=Mixer(0.8, 10, 50.0),
                  kpts={'size': (k, k, k), 'gamma': True},
                  xc=xc('TB09'),
                  convergence={'bands': -3},
                  txt='si.txt')
e = atoms.get_potential_energy()
gap, (sv, kv, nv), (sc, kc, nc) = bandgap(atoms.calc)
c = atoms.calc.hamiltonian.xc.c
print(gap, kv, kc)
print('c:', c)
assert abs(gap - 1.226) < 0.01
assert kv == 0 and kc == 24
assert abs(c - 1.136) < 0.01
