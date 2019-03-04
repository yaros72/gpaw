from __future__ import division, print_function

import numpy as np

from scipy.spatial import Delaunay, Voronoi, ConvexHull
from scipy.spatial.qhull import QhullError

from ase.dft.kpoints import monkhorst_pack

import gpaw.mpi as mpi
from gpaw import GPAW, restart
from gpaw.symmetry import Symmetry, aglomerate_points
from gpaw.kpt_descriptor import to1bz, kpts2sizeandoffsets
from itertools import product
from gpaw.mpi import world


def get_lattice_symmetry(cell_cv, tolerance=1e-7):
    """Return symmetry object of lattice group.

    Parameters
    ----------
    cell_cv : ndarray
        Unit cell.

    Returns
    -------
    gpaw.symmetry object

    """
    latsym = Symmetry([0], cell_cv, tolerance=tolerance)
    latsym.find_lattice_symmetry()
    return latsym


def find_high_symmetry_monkhorst_pack(calc, density,
                                      pbc=None):
    """Make high symmetry Monkhorst Pack k-point grid.

    Searches for and returns a Monkhorst Pack grid which
    contains the corners of the irreducible BZ so that when the
    number of kpoints are reduced the full irreducible brillouion
    zone is spanned.

    Parameters
    ----------
    calc : str
        The path to a calculator object.
    density : float
        The required minimum density of the Monkhorst Pack grid.
    pbc : Boolean list/ndarray of shape (3,) or None
        List indicating periodic directions. If None then
        pbc = [True] * 3.

    Returns
    -------
    ndarray
        Array of shape (nk, 3) containing the kpoints.

    """

    if pbc is None:
        pbc = np.array([True, True, True])
    else:
        pbc = np.array(pbc)

    atoms, calc = restart(calc, txt=None)
    minsize, offset = kpts2sizeandoffsets(density=density, even=True,
                                          gamma=True, atoms=atoms)

    bzk_kc, ibzk_kc, latibzk_kc = get_bz(calc, returnlatticeibz=True)

    maxsize = minsize + 10
    minsize[~pbc] = 1
    maxsize[~pbc] = 2

    if mpi.rank == 0:
        print('Brute force search for symmetry ' +
              'complying MP-grid... please wait.')

    for n1 in range(minsize[0], maxsize[0]):
        for n2 in range(minsize[1], maxsize[1]):
            for n3 in range(minsize[2], maxsize[2]):
                size = [n1, n2, n3]
                size, offset = kpts2sizeandoffsets(size=size, gamma=True,
                                                   atoms=atoms)

                ints = ((ibzk_kc + 0.5 - offset) * size - 0.5)[:, pbc]

                if (np.abs(ints - np.round(ints)) < 1e-5).all():
                    kpts_kc = monkhorst_pack(size) + offset
                    kpts_kc = to1bz(kpts_kc, calc.wfs.gd.cell_cv)

                    for ibzk_c in ibzk_kc:
                        diff_kc = np.abs(kpts_kc - ibzk_c)[:, pbc].round(6)
                        if not (np.mod(np.mod(diff_kc, 1), 1) <
                                1e-5).all(axis=1).any():
                            raise AssertionError('Did not find ' + str(ibzk_c))
                    if mpi.rank == 0:
                        print('Done. Monkhorst-Pack grid:', size, offset)
                    return kpts_kc

    if mpi.rank == 0:
        print('Did not find matching kpoints for the IBZ')
        print(ibzk_kc.round(5))

    raise RuntimeError


def unfold_points(points, U_scc, tol=1e-8, mod=None):
    """Unfold k-points using a given set of symmetry operators.

    Parameters
    ----------
    points: ndarray
    U_scc: ndarray
    tol: float
        Tolerance indicating when kpoints are considered to be
        identical.
    mod: integer 1 or None
        Consider kpoints spaced by a full reciprocal lattice vector
        to be identical.

    Returns
    -------
    ndarray
        Array of shape (nk, 3) containing the unfolded kpoints.
    """

    points = np.concatenate(np.dot(points, U_scc.transpose(0, 2, 1)))
    return unique_rows(points, tol=tol, mod=mod)


def unique_rows(ain, tol=1e-10, mod=None, aglomerate=True):
    """Return unique rows of a 2D ndarray.

    Parameters
    ----------
    ain : 2D ndarray
    tol : float
        Tolerance indicating when kpoints are considered to be
        identical.
    mod : integer 1 or None
        Consider kpoints spaced by a full reciprocal lattice vector
        to be identical.
    aglomerate : bool
        Aglomerate clusters of points before comparing.

    Returns
    -------
    2D ndarray
        Array containing only unique rows.
    """
    # Move to positive octant
    a = ain - ain.min(0)

    # First take modulus
    if mod is not None:
        a = np.mod(np.mod(a, mod), mod)

    # Round and take modulus again
    if aglomerate:
        aglomerate_points(a, tol)
    a = a.round(-np.log10(tol).astype(int))
    if mod is not None:
        a = np.mod(a, mod)

    # Now perform ordering
    order = np.lexsort(a.T)
    a = a[order]

    # Find unique rows
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(1)

    return ain[order][ui]


def get_smallest_Gvecs(cell_cv, n=5):
    """Find smallest reciprocal lattice vectors.

    Parameters
    ----------
    cell_cv : ndarray
        Unit cell.
    n : int
        Sampling along each crystal axis.

    Returns
    -------
    G_xv : ndarray
        Reciprocal lattice vectors in cartesian coordinates.
    N_xc : ndarray
        Reciprocal lattice vectors in crystal coordinates.

    """
    B_cv = 2.0 * np.pi * np.linalg.inv(cell_cv).T
    N_xc = np.indices((n, n, n)).reshape((3, n**3)).T - n // 2
    G_xv = np.dot(N_xc, B_cv)

    return G_xv, N_xc


def get_symmetry_operations(U_scc, time_reversal):
    """Return point symmetry operations."""

    if U_scc is None:
        U_scc = np.array([np.eye(3)])

    inv_cc = -np.eye(3, dtype=int)
    has_inversion = (U_scc == inv_cc).all(2).all(1).any()

    if has_inversion:
        time_reversal = False

    if time_reversal:
        Utmp_scc = np.concatenate([U_scc, -U_scc])
    else:
        Utmp_scc = U_scc

    return Utmp_scc


def get_ibz_vertices(cell_cv, U_scc=None, time_reversal=None,
                     origin_c=None):
    """Determine irreducible BZ.

    Parameters
    ----------
    cell_cv : ndarray
        Unit cell
    U_scc : ndarray
        Crystal symmetry operations.
    time_reversal : bool
        Use time reversal symmetry?

    Returns
    -------
    ibzk_kc : ndarray
        Vertices of the irreducible BZ.
    """
    # Choose an origin
    if origin_c is None:
        origin_c = np.array([0.12, 0.22, 0.21], float)
    else:
        assert (np.abs(origin_c) < 0.5).all()

    if U_scc is None:
        U_scc = np.array([np.eye(3)])

    if time_reversal is None:
        time_reversal = False

    Utmp_scc = get_symmetry_operations(U_scc, time_reversal)

    icell_cv = np.linalg.inv(cell_cv).T
    B_cv = icell_cv * 2 * np.pi
    A_cv = np.linalg.inv(B_cv).T

    # Map a random point around
    point_sc = np.dot(origin_c, Utmp_scc.transpose((0, 2, 1)))
    assert len(point_sc) == len(unique_rows(point_sc))
    point_sv = np.dot(point_sc, B_cv)

    # Translate the points
    n = 5
    G_xv, N_xc = get_smallest_Gvecs(cell_cv, n=n)
    G_xv = np.delete(G_xv, n**3 // 2, axis=0)

    # Mirror points in plane
    N_xv = G_xv / (((G_xv**2).sum(1))**0.5)[:, np.newaxis]

    tp_sxv = (point_sv[:, np.newaxis] - G_xv[np.newaxis] / 2.)
    delta_sxv = ((tp_sxv * N_xv[np.newaxis]).sum(2)[..., np.newaxis] *
                 N_xv[np.newaxis])
    points_xv = (point_sv[:, np.newaxis] - 2 * delta_sxv).reshape((-1, 3))
    points_xv = np.concatenate([point_sv, points_xv])
    try:
        voronoi = Voronoi(points_xv)
    except QhullError:
        return get_ibz_vertices(cell_cv, U_scc=U_scc,
                                time_reversal=time_reversal,
                                origin_c=origin_c + [0.01, -0.02, -0.01])

    ibzregions = voronoi.point_region[0:len(point_sv)]

    ibzregion = ibzregions[0]
    ibzk_kv = voronoi.vertices[voronoi.regions[ibzregion]]
    ibzk_kc = np.dot(ibzk_kv, A_cv.T)

    return ibzk_kc


def get_bz(calc, returnlatticeibz=False, pbc_c=np.ones(3, bool)):
    """Return the BZ and IBZ vertices.

    Parameters
    ----------
    calc : str, GPAW calc instance

    Returns
    -------
    bzk_kc : ndarray
        Vertices of BZ in crystal coordinates
    ibzk_kc : ndarray
        Vertices of IBZ in crystal coordinates

    """

    if isinstance(calc, str):
        calc = GPAW(calc, txt=None)
    cell_cv = calc.wfs.gd.cell_cv

    # Crystal symmetries
    symmetry = calc.wfs.kd.symmetry
    cU_scc = get_symmetry_operations(symmetry.op_scc,
                                     symmetry.time_reversal)

    return get_reduced_bz(cell_cv, cU_scc, False, returnlatticeibz,
                          pbc_c=pbc_c)


def get_reduced_bz(cell_cv, cU_scc, time_reversal, returnlatticeibz=False,
                   pbc_c=np.ones(3, bool), tolerance=1e-7):

    """Reduce the BZ using the crystal symmetries to obtain the IBZ.

    Parameters
    ----------
    cell_cv : ndarray
        Unit cell.
    cU_scc : ndarray
        Crystal symmetry operations.
    time_reversal : bool
        Switch for time reversal.
    pbc: bool or [bool, bool, bool]
        Periodic bcs
    """

    if time_reversal:
        cU_scc = get_symmetry_operations(cU_scc, time_reversal)

    # Lattice symmetries
    latsym = get_lattice_symmetry(cell_cv, tolerance=tolerance)
    lU_scc = get_symmetry_operations(latsym.op_scc,
                                     latsym.time_reversal)

    # Find Lattice IBZ
    ibzk_kc = get_ibz_vertices(cell_cv,
                               U_scc=latsym.op_scc,
                               time_reversal=latsym.time_reversal)
    latibzk_kc = ibzk_kc.copy()

    # Expand lattice IBZ to crystal IBZ
    ibzk_kc = expand_ibz(lU_scc, cU_scc, ibzk_kc, pbc_c=pbc_c)

    # Fold out to full BZ
    bzk_kc = unique_rows(np.concatenate(np.dot(ibzk_kc,
                                               cU_scc.transpose(0, 2, 1))))

    if returnlatticeibz:
        return bzk_kc, ibzk_kc, latibzk_kc
    else:
        return bzk_kc, ibzk_kc


def expand_ibz(lU_scc, cU_scc, ibzk_kc, pbc_c=np.ones(3, bool)):
    """Expand IBZ from lattice group to crystal group.

    Parameters
    ----------
    lU_scc : ndarray
        Lattice symmetry operators.
    cU_scc : ndarray
        Crystal symmetry operators.
    ibzk_kc : ndarray
        Vertices of lattice IBZ.

    Returns
    -------
    ibzk_kc : ndarray
        Vertices of crystal IBZ.

    """

    # Find right cosets. The lattice group is partioned into right cosets of
    # the crystal group. This can in practice be done by testing whether
    # U1 U2^{-1} is in the crystal group as done below.
    cosets = []
    Utmp_scc = lU_scc.copy()
    while len(Utmp_scc):
        U1_cc = Utmp_scc[0].copy()
        Utmp_scc = np.delete(Utmp_scc, 0, axis=0)
        j = 0
        new_coset = [U1_cc]
        while j < len(Utmp_scc):
            U2_cc = Utmp_scc[j]
            U3_cc = np.dot(U1_cc, np.linalg.inv(U2_cc))
            if (U3_cc == cU_scc).all(2).all(1).any():
                new_coset.append(U2_cc)
                Utmp_scc = np.delete(Utmp_scc, j, axis=0)
                j -= 1
            j += 1
        cosets.append(new_coset)

    volume = np.inf
    nibzk_kc = ibzk_kc
    U0_cc = cosets[0][0]  # Origin

    if np.any(~pbc_c):
        nonpbcind = np.argwhere(~pbc_c)

    # Once the coests are known the irreducible zone is given by picking one
    # operation from each coset. To make sure that the IBZ produced is simply
    # connected we compute the volume of the convex hull of the produced IBZ
    # and pick (one of) the ones that have the smallest volume. This is done by
    # brute force and can sometimes take a while, however, in most cases this
    # is not a problem.
    combs = list(product(*cosets[1:]))[world.rank::world.size]
    for U_scc in combs:
        if not len(U_scc):
            continue
        U_scc = np.concatenate([np.array(U_scc), [U0_cc]])
        tmpk_kc = unfold_points(ibzk_kc, U_scc)
        volumenew = convex_hull_volume(tmpk_kc)

        if np.any(~pbc_c):
            # Compute the area instead
            volumenew /= (tmpk_kc[:, nonpbcind].max() -
                          tmpk_kc[:, nonpbcind].min())

        if volumenew < volume:
            nibzk_kc = tmpk_kc
            volume = volumenew

    ibzk_kc = unique_rows(nibzk_kc)
    volume = np.array((volume,))

    volumes = np.zeros(world.size, float)
    world.all_gather(volume, volumes)

    minrank = np.argmin(volumes)
    minshape = np.array(ibzk_kc.shape)
    world.broadcast(minshape, minrank)

    if world.rank != minrank:
        ibzk_kc = np.zeros(minshape, float)
    world.broadcast(ibzk_kc, minrank)

    return ibzk_kc


def tetrahedron_volume(a, b, c, d):
    """Calculate volume of tetrahedron.

    Parameters
    ----------
    a, b, c, d : ndarray
        Vertices of tetrahedron.

    Returns
    -------
    float
        Volume of tetrahedron.

    """
    return np.abs(np.einsum('ij,ij->i', a - d,
                            np.cross(b - d, c - d))) / 6


def convex_hull_volume(pts):
    """Calculate volume of the convex hull of a collection of points.

    Parameters
    ----------
    pts : list, ndarray
        A list of 3d points.

    Returns
    -------
    float
        Volume of convex hull.

    """
    hull = ConvexHull(pts)
    dt = Delaunay(pts[hull.vertices])
    tets = dt.points[dt.simplices]
    vol = np.sum(tetrahedron_volume(tets[:, 0], tets[:, 1],
                                    tets[:, 2], tets[:, 3]))
    return vol


