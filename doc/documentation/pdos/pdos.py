# Creates: pdos.png
from gpaw import GPAW, restart
import matplotlib.pyplot as plt

# Density of States
plt.subplot(211)
slab, calc = restart('top.gpw')
e, dos = calc.get_dos(spin=0, npts=2001, width=0.2)
e_f = calc.get_fermi_level()
plt.plot(e - e_f, dos)
plt.axis([-15, 10, None, 4])
plt.ylabel('DOS')

molecule = range(len(slab))[-2:]

plt.subplot(212)
c_mol = GPAW('CO.gpw')
for n in range(2, 7):
    print('Band', n)
    # PDOS on the band n
    wf_k = [kpt.psit_nG[n] for kpt in c_mol.wfs.kpt_u]
    P_aui = [[kpt.P_ani[a][n] for kpt in c_mol.wfs.kpt_u]
             for a in range(len(molecule))]
    e, dos = calc.get_all_electron_ldos(mol=molecule, spin=0, npts=2001,
                                        width=0.2, wf_k=wf_k, P_aui=P_aui)
    plt.plot(e - e_f, dos, label='Band: ' + str(n))
plt.legend()
plt.axis([-15, 10, None, None])
plt.xlabel('Energy [eV]')
plt.ylabel('All-Electron PDOS')
plt.savefig('pdos.png')
plt.show()
