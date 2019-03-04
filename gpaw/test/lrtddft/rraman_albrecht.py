from ase import Atoms, Atom
from ase.vibrations.resonant_raman import ResonantRaman
from ase.vibrations.albrecht import Albrecht
from ase.parallel import world, DummyMPI

from gpaw import GPAW
from gpaw.lrtddft.kssingle import KSSingles
from gpaw.analyse.overlap import Overlap
from gpaw.test import equal

txt = '-'
txt = None
load = True
load = False
xc = 'LDA'

R = 0.7  # approx. experimental bond length
a = 4.0
c = 5.0
H2 = Atoms([Atom('H', (a / 2, a / 2, (c - R) / 2)),
            Atom('H', (a / 2, a / 2, (c + R) / 2))],
           cell=(a, a, c))

calc = GPAW(xc=xc, nbands=3, spinpol=False, eigensolver='rmm-diis', txt=txt)
H2.set_calculator(calc)
H2.get_potential_energy()

gsname = exname = 'rraman'
exkwargs={'eps':0.0, 'jend':1}
pz = ResonantRaman(H2, KSSingles, gsname=gsname, exname=exname,
                    exkwargs=exkwargs,
                   overlap=lambda x, y: Overlap(x).pseudo(y))
pz.run()

# check size
kss = KSSingles('rraman-d0.010.eq.ex.gz')
assert(len(kss) == 1)

om = 5

# Does Albrecht A work at all ?
# -----------------------------

al = Albrecht(H2, KSSingles, gsname=gsname, exname=exname,
              verbose=True, overlap=True)
ai = al.absolute_intensity(omega=om)[-1]
i = al.intensity(omega=om)[-1]

# parallel

if world.size > 1 and world.rank == 0:
    # single core
    comm = DummyMPI()
    pzsi = Albrecht(H2, KSSingles, gsname=gsname, exname=exname,
                   comm=comm, overlap=True, verbose=True)
    isi = pzsi.intensity(omega=om)[-1]
    equal(isi, i, 1e-11)

# Compare singles and multiples in Albrecht A
# -------------------------------------------

alas = Albrecht(H2, KSSingles, gsname=gsname, exname=exname,
              approximation='Albrecht A')
ints = alas.intensity(omega=om)[-1]
alam = Albrecht(H2, KSSingles, gsname=gsname, exname=exname,
                approximation='Albrecht A',
                skip=5, combinations=2)
# single excitation energies should agree
equal(ints, alam.intensity(omega=om)[0] , 1e-11)
alam.summary(omega=5, method='frederiksen')

