from gpaw import GPAW

G = [0.0, 0.0, 0.0]
L = [0.5, 0.0, 0.0]
F = [0.5, 0.5, 0.0]
Z = [0.5, 0.5, 0.5]

calc = GPAW('gs_Bi2Se3.gpw',
            kpts=[G, L, F, Z],
            symmetry='off',
            txt='high_sym.txt')
calc.diagonalize_full_hamiltonian(nbands=50)

calc.write('high_sym.gpw', mode='all')
