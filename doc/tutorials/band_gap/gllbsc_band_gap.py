from gpaw import GPAW, FermiDirac
from ase import Atoms

atom = Atoms(symbols='KTaO3',
             pbc=[ True,  True,  True],
             cell=[[ 4.,  0.,  0.],
                   [ 0.,  4.,  0.],
                   [ 0.,  0.,  4.]],
             positions=[[ 0.,  0.,  0.],
                        [ 2.,  2.,  2.],
                        [ 2.,  2.,  0.],
                        [ 0.,  2.,  2.],
                        [ 2.,  0.,  2.]])


calc = GPAW(h=0.16,
            kpts=(10,10,10),
            xc='GLLBSC',
            txt='KTaO3.out',
            occupations=FermiDirac(width=0.05),
            )

atom.set_calculator(calc)
atom.get_potential_energy()

#Important commands for calculating the response and the
#derivatice discontinuity
response = calc.hamiltonian.xc.xcs['RESPONSE']
response.calculate_delta_xc()
EKs, Dxc = response.calculate_delta_xc_perturbation()

# fundamental band gap
# EKs = kohn-sham bandgap
# Dxc = derivative discontinuity
Gap = EKs+Dxc

print("Calculated band gap:", Gap)
