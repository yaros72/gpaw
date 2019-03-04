from ase.build import molecule
from gpaw import GPAW
from gpaw.xc.libvdwxc import vdw_df_cx

atoms = molecule('H2O')
atoms.center(vacuum=3.0)
calc = GPAW(xc=vdw_df_cx())
atoms.set_calculator(calc)
atoms.get_potential_energy()
