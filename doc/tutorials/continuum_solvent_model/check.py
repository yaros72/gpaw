from ase.io import read
from ase.units import mol, kcal
# calculate solvation Gibbs energy in various units
Egasphase = read('gasphase.txt').get_potential_energy()
Ewater = read('water.txt').get_potential_energy()
DGSol_eV = Ewater - Egasphase
DGSol_kcal_per_mol = DGSol_eV / (kcal / mol)
assert abs(DGSol_kcal_per_mol - -4.5) < 0.05
