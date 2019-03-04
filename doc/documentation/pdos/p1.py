from gpaw import GPAW, restart
import pickle

slab, calc = restart('top.gpw')
c_mol = GPAW('CO.gpw')
molecule = range(len(slab))[-2:]
e_n = []
P_n = []
for n in range(c_mol.get_number_of_bands()):
    print('Band: ', n)
    wf_k = [kpt.psit_nG[n] for kpt in c_mol.wfs.kpt_u]
    P_aui = [[kpt.P_ani[a][n] for kpt in c_mol.wfs.kpt_u]
             for a in range(len(molecule))]
    e, P = calc.get_all_electron_ldos(mol=molecule, wf_k=wf_k, spin=0,
                                      P_aui=P_aui, raw=True)
    e_n.append(e)
    P_n.append(P)
pickle.dump((e_n, P_n), open('top.pickle', 'wb'))
