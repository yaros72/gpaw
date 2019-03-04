from ase.build import bulk
from gpaw import GPAW, PW
from gpaw.xc.libvdwxc import vdw_df_cx

# "Large" system:
atoms = bulk('Cu').repeat((2, 2, 2))
calc = GPAW(mode=PW(600),
            kpts=(4, 4, 4),
            xc=vdw_df_cx(mode='pfft', pfft_grid=(2, 2)),
            parallel=dict(kpt=4, augment_grids=True))
atoms.set_calculator(calc)
atoms.get_potential_energy()
