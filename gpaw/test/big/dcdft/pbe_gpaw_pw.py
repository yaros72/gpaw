import sys
import time

import numpy as np

import ase.db
from ase.units import Rydberg
from ase.utils import opencew
from ase.calculators.calculator import kpts2mp
from ase.io import Trajectory
from ase.test.tasks.dcdft import DeltaCodesDFTCollection as Collection
from gpaw import GPAW, PW, FermiDirac
from gpaw.utilities import h2gpts

collection = Collection()

if len(sys.argv) == 1:
    names = collection.names
else:
    names = [sys.argv[1]]

c = ase.db.connect('dcdft_gpaw_pw.db')

# mode = 'lcao'
mode = 'fd'
mode = 'pw'

e = 0.08  # h -> gpts
e = round(100 * Rydberg, 0)

kptdensity = 16.0  # this is converged
kptdensity = 6.0  # just for testing
width = 0.01

relativistic = True
constant_basis = True

if relativistic:
    linspace = (0.98, 1.02, 5)  # eos numpy's linspace
else:
    linspace = (0.92, 1.08, 7)  # eos numpy's linspace
linspacestr = ''.join([str(t) + 'x' for t in linspace])[:-1]

code = 'gpaw' + '-' + mode + str(e) + '_c' + str(constant_basis) + '_e' + linspacestr
code = code + '_k' + str(kptdensity) + '_w' + str(width)
code = code + '_r' + str(relativistic)

for name in names:
    # save all steps in one traj file in addition to the database
    # we should only used the database c.reserve, but here
    # traj file is used as another lock ...
    fd = opencew(name + '_' + code + '.traj')
    if fd is None:
        continue
    traj = Trajectory(name + '_' + code + '.traj', 'w')
    atoms = collection[name]
    cell = atoms.get_cell()
    kpts = tuple(kpts2mp(atoms, kptdensity, even=True))
    kwargs = {}
    if mode in ['fd', 'lcao']:
        if constant_basis:
            # gives more smooth EOS in fd mode
            kwargs.update({'gpts': h2gpts(e, cell)})
        else:
            kwargs.update({'h': e})
    elif mode == 'pw':
        if constant_basis:
            kwargs.update({'mode': PW(e, cell=cell)})
            kwargs.update({'gpts': h2gpts(0.10, cell)})
        else:
            kwargs.update({'mode': PW(e)})
    if mode == 'pw':
        if name in ['Li', 'Na']:
            # https://listserv.fysik.dtu.dk/pipermail/gpaw-developers/2012-May/002870.html
            if constant_basis:
                kwargs.update({'gpts': h2gpts(0.05, cell)})
            else:
                kwargs.update({'h': 0.05})
    if mode == 'lcao':
        kwargs.update({'mode': 'lcao'})
        kwargs.update({'basis': 'dzp'})
    if name in ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Ca', 'Sr', 'Ba', 'Be']:
        # results wrong / scf slow with minimal basis
        kwargs.update({'basis': 'dzp'})
        kwargs.update({'nbands': -5})
    # loop over EOS linspace
    for n, x in enumerate(np.linspace(linspace[0], linspace[1], linspace[2])):
        id = c.reserve(name=name, mode=mode, e=e, linspacestr=linspacestr,
                       kptdensity=kptdensity, width=width,
                       relativistic=relativistic,
                       constant_basis=constant_basis,
                       x=x)
        if id is None:
            continue
        # perform EOS step
        atoms.set_cell(cell * x, scale_atoms=True)
        # set calculator
        atoms.calc = GPAW(
            txt=name + '_' + code + '_' + str(n) + '.txt',
            xc='PBE',
            kpts=kpts,
            occupations=FermiDirac(width),
            parallel={'band': 1},
            maxiter=777,
            idiotproof=False)
        atoms.calc.set(**kwargs)  # remaining calc keywords
        t = time.time()
        atoms.get_potential_energy()
        c.write(atoms,
                name=name, mode=mode, e=e, linspacestr=linspacestr,
                kptdensity=kptdensity, width=width,
                relativistic=relativistic,
                constant_basis=constant_basis,
                x=x,
                niter=atoms.calc.get_number_of_iterations(),
                time=time.time()-t)
        traj.write(atoms)
        del c[id]
