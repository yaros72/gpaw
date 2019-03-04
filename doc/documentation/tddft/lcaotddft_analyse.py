import matplotlib.pyplot as plt
from ase.io import read
from gpaw.lcaotddft.tddfpt import transform_local_operator

transform_local_operator(gpw_file='Na8_gs.gpw',
                         tdop_file='Na8.TdDen',
                         fqop_file='Na8.FqDen',
                         omega=1.8,
                         eta=0.23)
dct = read('Na8.FqDen.imag.cube', full_output=True)
data = dct['data'][:, :, 16]
atoms = dct['atoms']
extent = [0, atoms.cell[0][0], 0, atoms.cell[1][1]]
plt.imshow(data.T, origin='lower', extent=extent)

for atom in atoms:
    circle = plt.Circle((atom.position[0], atom.position[1]),
                        0.3, color='r', clip_on=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle)

plt.title('Induced density of $Na_{8}$')
plt.xlabel('$\\AA$')
plt.ylabel('$\\AA$')
plt.savefig('Na8_imag.png')
