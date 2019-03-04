import numpy as np

from ase import Atoms
from ase.parallel import parprint
from ase.units import Hartree

import gpaw.mpi as mpi
from gpaw import GPAW
from gpaw.test import equal
from gpaw.lrtddft.kssingle import KSSingles


Be = Atoms('Be')
Be.center(vacuum=4)
if 1:
    # introduce a sligth non-orthgonality
    cell = Be.get_cell()
    cell[1] += 0.001 * cell[0]
    Be.set_cell(cell)

txt = None
#txt='-'
eigensolver = None
#eigensolver = 'rmm-diis'

#modes = ['lcao', 'fd']
modes = ['fd']

for mode in modes:
    energy = {}
    osz = {}
    for pbc in [False, True]:
        Be.set_pbc(pbc)
        if pbc:
            name = 'periodic'
            calc = GPAW(h=0.25, nbands=4, kpts=(1,2,2), mode=mode,
                        poissonsolver={'name': 'fd'},
                        symmetry='off',
                        eigensolver=eigensolver, txt=txt)
        else:
            name = 'zero_bc'
            calc = GPAW(h=0.25, nbands=4, mode=mode,
                        poissonsolver={'name': 'fd'},
                        eigensolver=eigensolver, txt=txt)
        Be.set_calculator(calc)
        Be.get_potential_energy()

        kss = KSSingles(calc, eps=0.9)
        # all s->p transitions at the same energy [Ha] and
        # oscillator_strength
        for ks in kss:
            equal(ks.get_energy(), kss[0].get_energy(), 5.e-3)
            equal(ks.get_oscillator_strength()[0],
                  kss[0].get_oscillator_strength()[0], 5.e-3)
            equal(ks.get_oscillator_strength()[0],
                  ks.get_oscillator_strength()[1:].sum() / 3, 1.e-15)
            for c in range(3):
                equal(ks.get_oscillator_strength()[1 + c],
                      ks.get_dipole_tensor()[c, c], 1.e-15)
        energy[name] = np.array(
            [ks.get_energy() * Hartree for ks in kss]).mean()
        osz[name] = np.array(
            [ks.get_oscillator_strength()[0] for ks in kss]).sum()

        parprint(name + ':')
        parprint(kss)

        # I/O
        fname = 'kss_' + name + '.dat'
        kss.write(fname)
        mpi.world.barrier()
        kss = KSSingles(fname)
        kss1 = KSSingles(fname, jend=1)
        assert(len(kss1) == calc.wfs.kd.nks)

    # periodic and non-periodic should be roughly equal
    equal(energy['zero_bc'], energy['periodic'], 5.e-2)
    equal(osz['zero_bc'], osz['periodic'], 2.e-2)
