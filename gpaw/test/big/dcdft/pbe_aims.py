import os
import sys
import time

import numpy as np

import ase.db
from ase.utils import opencew
from ase.calculators.calculator import kpts2mp
from ase.io import Trajectory
from ase.calculators.aims import Aims
from ase.test.tasks.dcdft import DeltaCodesDFTCollection as Collection

collection = Collection()

if len(sys.argv) == 1:
    names = collection.names
else:
    names = [sys.argv[1]]

c = ase.db.connect('dcdft_aims.db')

# select the basis set
basis = 'light'
#basis = 'tight'
#basis = 'really_tight'
#basis = 'tier2'

kptdensity = 16.0  # this is converged
kptdensity = 6.0  # just for testing
width = 0.01

basis_threshold = 0.00001
relativistic = 'none'
relativistic = 1.e-12
relativistic = 'scalar'

sc_accuracy_rho = 1.e-4
sc_accuracy_eev = 5.e-3

if relativistic == 'none':
    linspace = (0.92, 1.08, 7)  # eos numpy's linspace
else:
    linspace = (0.98, 1.02, 5)  # eos numpy's linspace
linspacestr = ''.join([str(t) + 'x' for t in linspace])[:-1]

code = 'aims' + '-' + basis + '_e' + linspacestr
code = code + '_k' + str(kptdensity) + '_w' + str(width)
code = code + '_t' + str(basis_threshold) + '_r' + str(relativistic)

collection = Collection()

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
    if relativistic == 'scalar':
        kwargs.update({'relativistic': ['atomic_zora', relativistic]})
    elif relativistic == 'none':
        kwargs.update({'relativistic': 'none'})
    else:  # e.g. 1.0e-12
        kwargs.update({'relativistic': ['zora', 'scalar', relativistic]})
    if atoms.get_initial_magnetic_moments().any():  # spin-polarization
        magmom = atoms.get_initial_magnetic_moments().sum() / len(atoms)
        kwargs.update({'spin': 'collinear'})
    # convergence problems for tier2
    charge_mix_param = 0.01
    basis_threshold = 0.00001
    if basis in ['tier2']:
        if name in ['Cr', 'Fe'] and relativistic == 'none':
            basis_threshold = 0.00005
            sc_accuracy_rho=2.5e-3
            sc_accuracy_eev=5.e-3
        if name in ['Mn']:
            charge_mix_param = 0.01
            basis_threshold = 0.00005
            sc_accuracy_rho=2.5e-3
            sc_accuracy_eev=5.e-3
            if relativistic == 'none':
                sc_accuracy_rho=3.0e-3
    # loop over EOS linspace
    for n, x in enumerate(np.linspace(linspace[0], linspace[1], linspace[2])):
        id = c.reserve(name=name, basis=basis, linspacestr=linspacestr,
                       kptdensity=kptdensity, width=width,
                       basis_threshold=basis_threshold,
                       relativistic=relativistic,
                       x=x)
        if id is None:
            continue
        # perform EOS step
        atoms.set_cell(cell * x, scale_atoms=True)
        # set calculator
        atoms.calc = Aims(
            label=name + '_' + code + '_' + str(n),
            species_dir=os.path.join(os.environ['AIMS_SPECIES_DIR'], basis),
            xc='PBE',
            kpts=kpts,
            KS_method='elpa',
            sc_accuracy_rho=sc_accuracy_rho,
            sc_accuracy_eev=sc_accuracy_eev,
            occupation_type=['gaussian', width],
            override_relativity=True,
            override_illconditioning=True,
            basis_threshold=basis_threshold,
            charge_mix_param=charge_mix_param,
            sc_iter_limit=9000,
            )
        atoms.calc.set(**kwargs)  # remaining calc keywords
        t = time.time()
        atoms.get_potential_energy()
        c.write(atoms,
                name=name, basis=basis, linspacestr=linspacestr,
                kptdensity=kptdensity, width=width,
                basis_threshold=basis_threshold,
                relativistic=relativistic,
                x=x,
                time=time.time()-t)
        traj.write(atoms)
        del c[id]
