from ase import Atoms
from gpaw import GPAW
from gpaw.lrtddft import LrTDDFT
from gpaw.lrtddft.spectrum import spectrum

txt = '-'
txt = None
load = True
load = False
xc = 'LDA'

R = 0.7  # approx. experimental bond length
a = 4.0
c = 5.0
H2 = Atoms('HH',
           [(a / 2, a / 2, (c - R) / 2),
            (a / 2, a / 2, (c + R) / 2)],
           cell=(a, a, c))

calc = GPAW(xc=xc, nbands=2, spinpol=False, eigensolver='rmm-diis', txt=txt)
H2.set_calculator(calc)
H2.get_potential_energy()
calc.write('H2saved_wfs.gpw', 'all')
calc.write('H2saved.gpw')
wfs_error = calc.wfs.eigensolver.error

print('-> starting directly after a gs calculation')
lr = LrTDDFT(calc, txt='-')
lr.diagonalize()

print('-> reading gs with wfs')
gs = GPAW('H2saved_wfs.gpw', txt=txt)

lr1 = LrTDDFT(gs, xc=xc, txt='-')
# check the oscillator strrength
assert (abs(lr1[0].get_oscillator_strength()[0] /
            lr[0].get_oscillator_strength()[0] - 1) < 1e-7)

print('-> reading gs without wfs')
gs = GPAW('H2saved.gpw', txt=None)

lr2 = LrTDDFT(gs, txt='-')
# check the oscillator strrength
d = abs(lr2[0].get_oscillator_strength()[0] /
        lr[0].get_oscillator_strength()[0] - 1)
assert (d < 2e-3), d

# write spectrum
spectrum(lr, 'spectrum.dat')
