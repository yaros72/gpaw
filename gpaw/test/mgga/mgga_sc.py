"""Compare TPSS from scratch and from PBE"""
from ase import Atoms
from gpaw import GPAW, MixerSum, Davidson

n = Atoms('N', magmoms=[3])
n.center(vacuum=2.5)
def getkwargs():
    return dict(eigensolver=Davidson(4), mixer=MixerSum(0.5, 5, 10.0))

n.calc = GPAW(xc='TPSS', **getkwargs())
e1 = n.get_potential_energy()

n.calc = GPAW(xc='PBE', **getkwargs())
n.get_potential_energy()
n.calc.set(xc='TPSS')
e2 = n.get_potential_energy()
err = abs(e2 - e1)
print('Energy difference', err)
assert err < 3e-5, err
