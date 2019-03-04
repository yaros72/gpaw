# Creates: Ag.png
from ase.build import bulk
from gpaw import GPAW

# Perform standard ground state calculation (with plane wave basis)
ag = bulk('Ag')
calc = GPAW(mode='pw',
            xc='GLLBSC',
            kpts=(10, 10, 10),
            txt='Ag_GLLBSC.txt')
ag.set_calculator(calc)
ag.get_potential_energy()
calc.write('Ag_GLLBSC.gpw')
ef = calc.get_fermi_level()

# Restart from ground state and fix potential:
calc = GPAW('Ag_GLLBSC.gpw',
            nbands=16,
            basis='dzp',
            fixdensity=True,
            symmetry='off',
            convergence={'bands': 12})

calc.set(kpts={'path': 'WLGXWK', 'npoints': 100})
calc.get_potential_energy()

# Plot the band structure
band_structure = calc.band_structure()
band_structure.plot(filename='Ag.png', emax=20.0, show=True)
