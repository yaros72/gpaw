import numpy as np
from ase.units import Bohr
from gpaw.grid_descriptor import GridDescriptor
from gpaw.utilities.grid import grid2grid


def extended_grid_descriptor(gd,
                             extend_N_cd=None,
                             N_c=None, extcomm=None):
    """Create grid descriptor for extended grid.

    Provide only either extend_N_cd or N_c.

    Parameters:

    extend_N_cd: ndarray, int
        Number of extra grid points per axis (c) and
        direction (d, left or right)
    N_c: ndarray, int
        Number of grid points in extended grid
    extcomm:
        Communicator for the extended grid, defaults to gd.comm
    """

    if extcomm is None:
        extcomm = gd.comm

    if extend_N_cd is None:
        assert N_c is not None, 'give only extend_N_cd or N_c'
        N_c = np.array(N_c, dtype=np.int)
        extend_N_cd = np.tile((N_c - gd.N_c) // 2, (2, 1)).T
    else:  # extend_N_cd is not None:
        assert N_c is None, 'give only extend_N_cd or N_c'
        extend_N_cd = np.array(extend_N_cd, dtype=np.int)
        N_c = gd.N_c + extend_N_cd.sum(axis=1)

    cell_cv = gd.h_cv * N_c
    move_c = gd.get_grid_spacings() * extend_N_cd[:, 0]

    egd = GridDescriptor(N_c, cell_cv, gd.pbc_c, extcomm)
    egd.extend_N_cd = extend_N_cd

    return egd, cell_cv * Bohr, move_c * Bohr


def extend_array(gd, ext_gd, a_g, aext_g):
    assert gd.comm.compare(ext_gd.comm) in ['ident', 'congruent']
    offset_c = ext_gd.extend_N_cd[:, 0]
    grid2grid(gd.comm, gd, ext_gd, a_g, aext_g, offset1_c=offset_c)


def deextend_array(gd, ext_gd, a_g, aext_g):
    assert gd.comm.compare(ext_gd.comm) in ['ident', 'congruent']
    offset_c = ext_gd.extend_N_cd[:, 0]
    grid2grid(gd.comm, ext_gd, gd, aext_g, a_g, offset2_c=offset_c)


def move_atoms(atoms, move_c):
    pos_a = atoms.get_positions()
    for pos in pos_a:
        pos += move_c
    atoms.set_positions(pos_a)
