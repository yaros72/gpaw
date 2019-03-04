import json
from os.path import exists, splitext, isfile
from os import remove
from glob import glob

import numpy as np
from gpaw import GPAW
from gpaw.mpi import world
from gpaw.berryphase import get_polarization_phase
from ase.parallel import paropen
from ase.units import Bohr
from ase.io import jsonio


def get_wavefunctions(atoms, name, params):
    params['symmetry'] = {'point_group': False,
                          'do_not_symmetrize_the_density': True,
                          'time_reversal': False}
    tmp = splitext(name)[0]
    atoms.calc = GPAW(txt=tmp + '.txt', **params)
    atoms.get_potential_energy()
    atoms.calc.write(name, 'all')
    return atoms.calc


def borncharges(calc, delta=0.01):

    params = calc.parameters
    atoms = calc.atoms
    cell_cv = atoms.get_cell() / Bohr
    vol = abs(np.linalg.det(cell_cv))
    sym_a = atoms.get_chemical_symbols()

    Z_a = []
    for num in calc.atoms.get_atomic_numbers():
        for ida, setup in zip(calc.wfs.setups.id_a,
                              calc.wfs.setups):
            if abs(ida[0] - num) < 1e-5:
                break
        Z_a.append(setup.Nv)
    Z_a = np.array(Z_a)

    # List for atomic indices
    indices = list(range(len(sym_a)))

    pos_av = atoms.get_positions()
    avg_v = np.sum(pos_av, axis=0) / len(pos_av)
    pos_av -= avg_v
    atoms.set_positions(pos_av)
    Z_avv = []
    norm_c = np.linalg.norm(cell_cv, axis=1)
    proj_cv = cell_cv / norm_c[:, np.newaxis]

    B_cv = np.linalg.inv(cell_cv).T * 2 * np.pi
    area_c = np.zeros((3,), float)
    area_c[[2, 1, 0]] = [np.linalg.norm(np.cross(B_cv[i], B_cv[j]))
                         for i in range(3)
                         for j in range(3) if i < j]

    if world.rank == 0:
        print('Atomnum Atom Direction Displacement')
    for a in indices:
        phase_scv = np.zeros((2, 3, 3), float)
        for v in range(3):
            for s, sign in enumerate([-1, 1]):
                if world.rank == 0:
                    print(sym_a[a], a, v, s)
                # Update atomic positions
                atoms.positions = pos_av
                atoms.positions[a, v] = pos_av[a, v] + sign * delta
                prefix = 'born-{}-{}{}{}'.format(delta, a,
                                                 'xyz'[v],
                                                 ' +-'[sign])
                name = prefix + '.gpw'
                berryname = prefix + '-berryphases.json'
                if not exists(name) and not exists(berryname):
                    calc = get_wavefunctions(atoms, name, params)

                try:
                    phase_c = get_polarization_phase(name)
                except ValueError:
                    calc = get_wavefunctions(atoms, name, params)
                    phase_c = get_polarization_phase(name)

                phase_scv[s, :, v] = phase_c

                if exists(berryname):  # Calculation done?
                    if world.rank == 0:
                        # Remove gpw file
                        if isfile(name):
                            remove(name)

        dphase_cv = (phase_scv[1] - phase_scv[0])
        dphase_cv -= np.round(dphase_cv / (2 * np.pi)) * 2 * np.pi
        dP_cv = (area_c[:, np.newaxis] / (2 * np.pi)**3 *
                 dphase_cv)
        dP_vv = np.dot(proj_cv.T, dP_cv)
        Z_vv = dP_vv * vol / (2 * delta / Bohr)
        Z_avv.append(Z_vv)

    data = {'Z_avv': Z_avv, 'indices_a': indices, 'sym_a': sym_a}

    filename = 'borncharges-{}.json'.format(delta)

    with paropen(filename, 'w') as fd:
        json.dump(jsonio.encode(data), fd)

    world.barrier()
    if world.rank == 0:
        files = glob('born-*.gpw')
        for f in files:
            if isfile(f):
                remove(f)
