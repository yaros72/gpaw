from gpaw import GPAW

calc = GPAW('Pt_gs.gpw', txt=None)

calc = GPAW('Pt_gs.gpw',
            kpts={'path': 'GXWLGKX', 'npoints': 200},
            symmetry='off',
            fixdensity=True,
            txt='Pt_bands.txt')
calc.get_potential_energy()
calc.write('Pt_bands.gpw')
