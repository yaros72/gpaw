from gpaw import GPAW
from gpaw.lcaotddft.ksdecomposition import KohnShamDecomposition

# Calculate ground state with full unoccupied space
calc = GPAW('gs.gpw', nbands='nao', fixdensity=True,
            txt='unocc.out')
atoms = calc.get_atoms()
energy = atoms.get_potential_energy()
calc.write('unocc.gpw', mode='all')

# Construct KS electron-hole basis
ksd = KohnShamDecomposition(calc)
ksd.initialize(calc)
ksd.write('ksd.ulm')
