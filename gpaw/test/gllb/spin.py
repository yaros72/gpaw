from __future__ import print_function
from gpaw import GPAW, FermiDirac
from gpaw.test import equal
from ase.build import bulk


for spin in [False, True]:
    a = 3.56
    atoms = bulk('C', 'diamond', a=a)
    calc = GPAW(kpts=(3,3,3),
                xc='GLLBSC',
                spinpol = spin,
                nbands=8,
                convergence={'bands':6,'density':1e-6},
                occupations=FermiDirac(width=0.005))
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    calc.write('temp.gpw')
    response = calc.hamiltonian.xc.xcs['RESPONSE']
    response.calculate_delta_xc()
    #Eks is the Kohn-Sham gap and Dxc is the derivative discontinuity
    if spin:
        (Eksa, Dxca), (Eksb, Dxcb) = response.calculate_delta_xc_perturbation()
        Gapa = Eksa+Dxca
        Gapb = Eksb+Dxcb
        print("GAP", spin, Gapa, Gapb)
    else:
        Eks, Dxc = response.calculate_delta_xc_perturbation()
        Gap = Eks + Dxc
        print("GAP", spin, Gap)

equal(Gapa, Gapb, 1e-4)
equal(Gapa, Gap, 1e-4)
