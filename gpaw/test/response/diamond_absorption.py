import numpy as np
from ase.units import Bohr
from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.response.df import DielectricFunction
from gpaw.test import equal, findpeak

a = 6.75 * Bohr
atoms = bulk('C', 'diamond', a=a)

calc = GPAW(mode='pw',
            kpts=(3, 3, 3),
            eigensolver='rmm-diis',
            occupations=FermiDirac(0.001))

atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write('C.gpw', 'all')

eM1_ = 9.727
eM2_ = 9.548
w0_ = 10.7782
I0_ = 5.5472
w_ = 10.7532
I_ = 6.0686

# Macroscopic dielectric constant calculation
df = DielectricFunction('C.gpw', frequencies=(0.,), eta=0.001, ecut=50,
                        hilbert=False)
eM1, eM2 = df.get_macroscopic_dielectric_constant()
equal(eM1, eM1_, 0.01)
equal(eM2, eM2_, 0.01)

# Absorption spectrum calculation RPA
df = DielectricFunction('C.gpw', eta=0.25, ecut=50,
                        frequencies=np.linspace(0, 24., 241), hilbert=False)
a0, a = df.get_dielectric_function(filename=None)
df.check_sum_rule(a.imag)

equal(a0[0].real, eM1_, 0.01)
equal(a[0].real, eM2_, 0.01)
w, I = findpeak(np.linspace(0, 24., 241), a0.imag)
equal(w, w0_, 0.01)
equal(I / (4 * np.pi), I0_, 0.05)
w, I = findpeak(np.linspace(0, 24., 241), a.imag)
equal(w, w_, 0.01)
equal(I / (4 * np.pi), I_, 0.05)

a0, a = df.get_polarizability(filename=None)

w, I = findpeak(np.linspace(0, 24., 241), a0.imag)
equal(w, w0_, 0.01)
equal(I, I0_, 0.01)
w, I = findpeak(np.linspace(0, 24., 241), a.imag)
equal(w, w_, 0.01)
equal(I, I_, 0.01)

# Absorption spectrum calculation ALDA
w0_ = 10.7931
I0_ = 5.4131
w_ = 10.7562
I_ = 5.8803

a0, a = df.get_polarizability(filename=None, xc='ALDA')

w, I = findpeak(np.linspace(0, 24., 241), a0.imag)
equal(w, w0_, 0.01)
equal(I, I0_, 0.01)
w, I = findpeak(np.linspace(0, 24., 241), a.imag)
equal(w, w_, 0.01)
equal(I, I_, 0.01)

# Absorption spectrum calculation long-range kernel
w0_ = 10.2189
I0_ = 5.2061
w_ = 10.2906
I_ = 5.6955

a0, a = df.get_polarizability(filename=None, xc='LR0.25')

w, I = findpeak(np.linspace(0, 24., 241), a0.imag)
equal(w, w0_, 0.01)
equal(I, I0_, 0.01)
w, I = findpeak(np.linspace(0, 24., 241), a.imag)
equal(w, w_, 0.01)
equal(I, I_, 0.01)

# Absorption spectrum calculation Bootstrap
w0_ = 10.3800
I0_ = 5.3795
w_ = 10.4600
I_ = 6.0263 

a0, a = df.get_polarizability(filename=None, xc='Bootstrap')

w, I = findpeak(np.linspace(0, 24., 241), a0.imag)
equal(w, w0_, 0.01)
equal(I, I0_, 0.01)
w, I = findpeak(np.linspace(0, 24., 241), a.imag)
equal(w, w_, 0.01)
equal(I, I_, 0.01)

# Absorption spectrum calculation RPA - Wigner-Seitz truncation
w0_ = 10.7780
I0_ = 5.5472
w_ = 10.7532
I_ = 6.0750

df_ws = DielectricFunction('C.gpw', eta=0.25, ecut=50,
                           frequencies=np.linspace(0, 24., 241), hilbert=False,
                           truncation='wigner-seitz')

a0_ws, a_ws = df_ws.get_polarizability(filename=None)

w, I = findpeak(np.linspace(0, 24., 241), a0_ws.imag)
equal(w, w0_, 0.01)
equal(I, I0_, 0.01)
w, I = findpeak(np.linspace(0, 24., 241), a_ws.imag)
equal(w, w_, 0.1)
equal(I, I_, 0.1)
# The Wigner-Seitz truncation does not give exactly the same for kpts=(3,3,3)
