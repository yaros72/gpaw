from gpaw.solvation.cavity import get_pbc_positions
from ase.build import molecule
from ase.units import Bohr
import numpy as np


def get_nrepeats(nrepeats_c):
    return np.prod(np.array(nrepeats_c) * 2 + 1)


def check_valid_repeat(pos_v, rpos_v, nrepeats_c, cell_cv):
    nrepeats_c = np.array(nrepeats_c)
    diff_v = rpos_v - pos_v
    actual_nrepeats_c = np.dot(diff_v, np.linalg.inv(cell_cv))
    r_actual_nrepeats_c = np.around(actual_nrepeats_c)
    # check for integer multiple
    assert np.allclose(r_actual_nrepeats_c, actual_nrepeats_c)
    actual_nrepeats_c = np.abs(r_actual_nrepeats_c.astype(int))
    # check for too large repeats
    assert (actual_nrepeats_c <= nrepeats_c).all()


def check(pos_av, rpos_aav, nrepeats_c, cell_cv):
    assert len(pos_av) == len(rpos_aav)
    total_repeats = get_nrepeats(nrepeats_c)
    for a, pos_v in enumerate(pos_av):
        rpos_av = rpos_aav[a]
        assert len(rpos_av) == total_repeats
        for rpos_v in rpos_av:
            check_valid_repeat(pos_v, rpos_v, nrepeats_c, cell_cv)


cells = (
    ((10., .0, .0), (.0, 12., .0), (.0, .0, 12.)),  # orthogonal
    ((10., 1., .0), (.0, 12., 1.), (1., .0, 11.5)),  # non-orthogonal
    )

for cell in cells:
    atoms = molecule('H2O')
    atoms.center(vacuum=5.)
    atoms.set_cell(cell)
    pos_av = atoms.positions / Bohr
    cell_cv = atoms.get_cell() / Bohr
    Rcell_c = np.sqrt(np.sum(cell_cv ** 2, axis=1))

    # non periodic should never repeat
    atoms.pbc = np.zeros((3, ), dtype=bool)
    pos_aav = get_pbc_positions(atoms, 1e10 / Bohr)
    check(pos_av, pos_aav, (0, 0, 0), cell_cv)

    # periodic, zero cutoff should not repeat
    atoms.pbc = np.ones((3, ), dtype=bool)
    pos_aav = get_pbc_positions(atoms, .0 / Bohr)
    check(pos_av, pos_aav, (0, 0, 0), cell_cv)

    # periodic, cutoff <= cell size should repeat once
    atoms.pbc = np.ones((3, ), dtype=bool)
    pos_aav = get_pbc_positions(atoms, 0.99 * Rcell_c.min())
    check(pos_av, pos_aav, (1, 1, 1), cell_cv)

    # periodic, cutoff > cell size should repeat twice
    atoms.pbc = np.ones((3, ), dtype=bool)
    pos_aav = get_pbc_positions(atoms, 1.01 * Rcell_c.max())
    check(pos_av, pos_aav, (2, 2, 2), cell_cv)

    # mixed bc, cutoff > 2 * cell size should repeat three times
    for i in range(8):
        pbc = np.array([int(p) for p in np.binary_repr(i, 3)])
        atoms.pbc = pbc
        pos_aav = get_pbc_positions(atoms, 2.01 * Rcell_c.max())
        check(pos_av, pos_aav, pbc * 3, cell_cv)
