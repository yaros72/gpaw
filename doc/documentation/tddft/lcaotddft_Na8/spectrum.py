from gpaw.tddft.spectrum import photoabsorption_spectrum
photoabsorption_spectrum('dm.dat', 'spec.dat',
                         folding='Gauss', width=0.1,
                         e_min=0.0, e_max=10.0, delta_e=0.01)
