import os
import sys
import time

import numpy as np

import ase.db
from ase.units import Rydberg
from ase.utils import opencew
from ase.calculators.calculator import kpts2mp
from ase.io import Trajectory
from ase.calculators.abinit import Abinit
from ase.test.tasks.dcdft import DeltaCodesDFTCollection as Collection

collection = Collection()

if len(sys.argv) == 1:
    names = collection.names
else:
    names = [sys.argv[1]]

c = ase.db.connect('dcdft_abinit_paw.db')

ecut = 100
pawecutdg = 300

kptdensity = 16.0  # this is converged
kptdensity = 6.0  # just for testing
width = 0.01
ecutsm = 0.0
fband = 1.5
tolsym = 1.e-12

linspace = (0.98, 1.02, 5)  # eos numpy's linspace
linspacestr = ''.join([str(t) + 'x' for t in linspace])[:-1]

code = 'abinit' + '-' + '_c' + str(ecut) + str(pawecutdg) + '_e' + linspacestr
code = code + '_k' + str(kptdensity) + '_w' + str(width) + '_s' + str(ecutsm) + '_t' + str(tolsym)

for name in names:
    # save all steps in one traj file in addition to the database
    # we should only used the database c.reserve, but here
    # traj file is used as another lock ...
    fd = opencew(name + '_' + code + '.traj')
    if fd is None:
        continue
    traj = Trajectory(name + '_' + code + '.traj', 'w')
    atoms = collection[name]
    if name == 'Mn':  # fails to find the right magnetic state
        atoms.set_initial_magnetic_moments([10., 20., -10., -20.])
    if name == 'Co':  # fails to find the right magnetic state
        atoms.set_initial_magnetic_moments([10., 10.])
    if name == 'Ni':  # fails to find the right magnetic state
        atoms.set_initial_magnetic_moments([10., 10., 10., 10.])
    cell = atoms.get_cell()
    kpts = tuple(kpts2mp(atoms, kptdensity, even=True))
    kwargs = {}
    # loop over EOS linspace
    for n, x in enumerate(np.linspace(linspace[0], linspace[1], linspace[2])):
        id = c.reserve(name=name, ecut=ecut, pawecutdg=pawecutdg,
                       linspacestr=linspacestr,
                       kptdensity=kptdensity, width=width, ecutsm=ecutsm,
                       fband=fband, tolsym=tolsym,
                       x=x)
        if id is None:
            continue
        # perform EOS step
        atoms.set_cell(cell * x, scale_atoms=True)
        # set calculator
        atoms.calc = Abinit(
            pps='paw',  # uses highest valence
            label=name + '_' + code + '_' + str(n),
            xc='PBE',
            kpts=kpts,
            ecut=ecut*Rydberg,
            pawecutdg=pawecutdg*Rydberg,
            occopt=3,
            tsmear=width,
            ecutsm=ecutsm,
            toldfe=1.0e-6,
            nstep=900,
            pawovlp=-1,  # bypass overlap check
            fband=fband,
            # http://forum.abinit.org/viewtopic.php?f=8&t=35
            chksymbreak=0,
            tolsym=tolsym,
            prtwf=0,
            prtden=0,
        )
        atoms.calc.set(**kwargs)  # remaining calc keywords
        t = time.time()
        atoms.get_potential_energy()
        c.write(atoms,
                name=name, ecut=ecut, pawecutdg=pawecutdg,
                linspacestr=linspacestr,
                kptdensity=kptdensity, width=width, ecutsm=ecutsm,
                fband=fband, tolsym=tolsym,
                x=x,
                time=time.time()-t)
        traj.write(atoms)
        wfk = name + '_' + code + '_' + str(n) + 'o_WFK'
        if os.path.exists(wfk): os.remove(wfk)
        del c[id]
