from __future__ import print_function

from ase.build import bulk

from gpaw import GPAW, Davidson, Mixer, PW
from gpaw.xc.libvdwxc import vdw_mbeef

from gpaw.test import gen

setup = gen('Si', xcname='PBEsol')

system = bulk('Si')
calc = GPAW(mode=PW(200), xc=vdw_mbeef(),
            kpts=(2, 2, 2),
            nbands=4,
            convergence=dict(density=1e-6),
            mixer=Mixer(1.0),
            eigensolver=Davidson(4),
            setups={'Si': setup})
system.calc = calc
e = system.get_potential_energy()

ref = -60.557653422801671
err = abs(e - ref)
print('e=%r ref=%r err=%r' % (e, ref, err))
assert err < 1e-6, err
