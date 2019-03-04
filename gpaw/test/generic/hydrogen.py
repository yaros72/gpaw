from math import log
from ase import Atoms
from ase.io import read, write, iread
from ase.units import Bohr
from ase.db import connect
from gpaw import GPAW, FermiDirac
from gpaw.test import equal

a = 4.0
h = 0.2
hydrogen = Atoms('H',
                 [(a / 2, a / 2, a / 2)],
                 cell=(a, a, a))

hydrogen.calc = GPAW(h=h,
                     nbands=1,
                     convergence={'energy': 1e-7},
                     txt='h.txt')
e1 = hydrogen.get_potential_energy()
equal(e1, 0.526939, 0.001)

dens = hydrogen.calc.density
c = dens.gd.find_center(dens.nt_sG[0]) * Bohr
equal(abs(c - a / 2).max(), 0, 1e-13)

kT = 0.001
hydrogen.calc.set(occupations=FermiDirac(width=kT))
e2 = hydrogen.get_potential_energy()
equal(e1, e2 + log(2) * kT, 3.0e-7)

# Test ase.db a bit:
for name in ['h.json', 'h.db']:
    con = connect(name, append=False)
    con.write(hydrogen)
    id = con.write(hydrogen, foo='bar', data={'abc': [1, 2, 3]})
    assert id == 2
    assert con.reserve(foo='bar') is None
    row = con.get(foo='bar')
    assert row.energy == e2
    assert sum(row.data.abc) == 6
    del con[1]
    assert con.reserve(x=42) == 3

    write('x' + name, hydrogen)
    write('xx' + name, [hydrogen, hydrogen])

    assert read(name + '@foo=bar')[0].get_potential_energy() == e2
    for n, h in zip([1, 0], iread(name + '@:')):
        assert n == len(h)

# Test parsing of GPAW's text output:
h = read('h.txt')
error = abs(h.calc.get_eigenvalues() - hydrogen.calc.get_eigenvalues()).max()
assert error < 1e-5, error

# Test get_electrostatic_potential() method
v = hydrogen.calc.get_electrostatic_potential()
print(v.shape, v.ptp())
