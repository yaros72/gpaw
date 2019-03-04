from ase.build import mx2
from ase.dft.kpoints import bandpath, special_points

from gpaw import GPAW
from gpaw.unfold import Unfold, find_K_from_k

a = 3.184
PC = mx2(a=a).get_cell(complete=True)
path = [special_points['hexagonal'][k] for k in 'MKG']
kpts, x, X = bandpath(path, PC, 48)
    
M = [[3, 0, 0], [0, 3, 0], [0, 0, 1]]

Kpts = []
for k in kpts:
    K = find_K_from_k(k, M)[0]
    Kpts.append(K)

calc_bands = GPAW('gs_3x3_defect.gpw',
                  fixdensity=True,
                  kpts=Kpts,
                  symmetry='off',
                  nbands=220,
                  convergence={'bands': 200})

calc_bands.get_potential_energy()
calc_bands.write('bands_3x3_defect.gpw', 'all')

unfold = Unfold(name='3x3_defect',
                calc='bands_3x3_defect.gpw',
                M=M,
                spinorbit=False)

unfold.spectral_function(kpts=kpts, x=x, X=X,
                         points_name=['M', 'K', 'G'])
