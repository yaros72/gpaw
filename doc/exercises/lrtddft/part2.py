from gpaw.lrtddft import LrTDDFT, photoabsorption_spectrum
lr = LrTDDFT(filename='Omega_Na2.gz')
lr.diagonalize()
lr.write('excitations_Na2.gz')
lr = LrTDDFT(filename='excitations_Na2.gz')
photoabsorption_spectrum(lr, 'Na2_spectrum.dat', e_min=0.0, e_max=10)
