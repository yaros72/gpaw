
from ase.build import molecule
from gpaw import GPAW
from gpaw import dscf

# Ground state calculation
calc = GPAW(nbands=8,
            h=0.2,
            xc='PBE',
            spinpol=True,
            convergence={'energy': 100,
                         'density': 100,
                         'eigenstates': 1.0e-9,
                         'bands': -1})

CO = molecule('CO')
CO.center(vacuum=3)
CO.set_calculator(calc)

E_gs = CO.get_potential_energy()

# Obtain the pseudowavefunctions and projector overlaps of the
# state which is to be occupied. n=5,6 is the 2pix and 2piy orbitals
n = 5
molecule = [0, 1]
wf_u = [kpt.psit_nG[n] for kpt in calc.wfs.kpt_u]
p_uai = [dict([(molecule[a], P_ni[n]) for a, P_ni in kpt.P_ani.items()])
         for kpt in calc.wfs.kpt_u]

# Excited state calculation
calc_es = GPAW(nbands=8,
               h=0.2,
               xc='PBE',
               spinpol=True,
               convergence={'energy': 100,
                            'density': 100,
                            'eigenstates': 1.0e-9,
                            'bands': -1})

CO.set_calculator(calc_es)
lumo = dscf.AEOrbital(calc_es, wf_u, p_uai)
# lumo = dscf.MolecularOrbital(calc, weights={0: [0, 0, 0,  1],
#                                             1: [0, 0, 0, -1]})
dscf.dscf_calculation(calc_es, [[1.0, lumo, 1]], CO)

E_es = CO.get_potential_energy()

print('Excitation energy: ', E_es - E_gs)
