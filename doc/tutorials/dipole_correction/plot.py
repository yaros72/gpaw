# Creates: zero.png, periodic.png, corrected.png, pwcorrected.png, slab.png
import numpy as np
import matplotlib.pyplot as plt
from ase.io import write
from gpaw import GPAW

# this test requires OpenEXR-libs

for name in ['zero', 'periodic', 'corrected','pwcorrected']:
    calc = GPAW(name + '.gpw', txt=None)

    efermi = calc.get_fermi_level()

    v = calc.get_electrostatic_potential().mean(0).mean(0)
    z = np.linspace(0, calc.atoms.cell[2, 2], len(v), endpoint=False)

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(z, v, label='xy-averaged potential')
    plt.plot([0, z[-1]], [efermi, efermi], label='Fermi level')

    if name.endswith('corrected'):
        n = 6  # get the vacuum level 6 grid-points from the boundary
        plt.plot([0.2, 0.2], [efermi, v[n]], 'r:')
        plt.text(0.23, (efermi + v[n]) / 2,
                 '$\phi$ = %.2f eV' % (v[n] - efermi), va='center')
        plt.plot([z[-1] - 0.2, z[-1] - 0.2], [efermi, v[-n]], 'r:')
        plt.text(z[-1] - 0.23, (efermi + v[-n]) / 2,
                 '$\phi$ = %.2f eV' % (v[-n] - efermi),
                 va='center', ha='right')

    plt.xlabel('$z$, $\AA$')
    plt.ylabel('(Pseudo) electrostatic potential, V')
    plt.xlim([0., z[-1]])
    if name == 'pwcorrected':
        title = 'PW-mode corrected'
    else:
        title = name.title()
    plt.title(title + ' boundary conditions')
    plt.savefig(name + '.png')

write('slab.pov', calc.atoms,
      rotation='-90x',
      show_unit_cell=2,
      transparent=False,
      display=False,
      run_povray=True)
