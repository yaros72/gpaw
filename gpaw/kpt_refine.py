# Copyright (C) 2016 R. Warmbier Materials for Energy Research Group,
# Wits University
from __future__ import division

import copy
import numpy as np
from ase.dft.kpoints import monkhorst_pack, get_monkhorst_pack_size_and_offset
from gpaw import KPointError
from gpaw.kpt_descriptor import KPointDescriptor

"""
This file provides routines to create non-uniform k-point grids. We use
locally uniform grids of different densities, practically grid refinement.

One starts from a standard Monkhorst-Pack grid and chooses a number of grid
points, which shall be replaced with a finer grid of a given size.

The user can further choose, whether the original symmetry of the system is
enforced (careful!) or whether a reduced symmetry should be used.

Optionally, the user can ask to add all k+q points, if not already included.

Please cite https://doi.org/10.1016/j.cpc.2018.03.001

Example (Graphene):

calc = GPAW(mode=PW(ecut=450),
            kpts={"size":[15,15,1], "gamma":True},
            kpt_refine={"center":[1./3,1./3,0.], "size":[3,3,1],
                        "reduce_symmetry":False,
                        "q":[1./15,1./15,0.]},
           )

"""


def create_kpoint_descriptor_with_refinement(refine, bzkpts_kc, nspins, atoms,
                                             symmetry, comm, timer):
    """Main routine to build refined k-point grids."""
    if 'center' not in refine:
        raise RuntimeError('Center for refinement not given!')
    if 'size' not in refine:
        raise RuntimeError('Grid size for refinement not given!')

    center_ic = np.array(refine.get('center'), dtype=float, ndmin=2)
    size = np.array(refine.get('size'), ndmin=2)
    reduce_symmetry = refine.get('reduce_symmetry', True)

    # Check that all sizes are odd. That's not so much an issue really.
    # But even Monkhorst-Pack grids have points on the boundary,
    # which just would require more special casing, which I want to avoid.
    if (np.array(size) % 2 == 0).any():
        raise RuntimeError('Grid size for refinement must be odd!  Is: {}'.
                           format(size))

    # Arguments needed for k-point descriptor construction
    kwargs = {'nspins': nspins, 'atoms': atoms, 'symmetry': symmetry,
              'comm': comm}

    # Define coarse grid points
    bzk_coarse_kc = bzkpts_kc

    # Define fine grid points
    centers_i, bzk_fine_kc, weight_fine_k = get_fine_bzkpts(center_ic, size,
                                                            bzk_coarse_kc,
                                                            kwargs)

    if reduce_symmetry:
        # Define new symmetry object ignoring symmetries violated by the
        # refined kpoints
        kd_fine = create_kpoint_descriptor(bzk_fine_kc, **kwargs)
        symm = prune_symmetries_kpoints(kd_fine, symmetry)
        del kd_fine
    else:
        symm = copy.copy(symmetry)
    kwargs['symmetry'] = symm

    # Create new descriptor with both sets of points

    with timer('Create mixed descriptor'):
        kd = create_mixed_kpoint_descriptor(bzk_coarse_kc,
                                            bzk_fine_kc,
                                            centers_i,
                                            weight_fine_k,
                                            kwargs)

    # Add missing k-points to fulfill group properties with
    # zero-weighted points
    with timer('Add_missing_points'):
        kd = add_missing_points(kd, kwargs)

    # Add additional +q k-points, if necessary
    if 'q' in refine:
        timer.start("+q")
        N_coarse_c = get_monkhorst_pack_size_and_offset(bzk_coarse_kc)[0]
        bla = N_coarse_c * refine['q']
        if not max(abs(bla - np.rint(bla))) < 1e-8:
            kd.refine_info.almostoptical = True
        kd = add_plusq_points(kd, refine['q'], kwargs)
        symm = kd.symmetry
        kwargs['symmetry'] = symm
        kd = add_missing_points(kd, kwargs)
        timer.stop("+q")

    return kd


def create_kpoint_descriptor(bzkpts_kc, nspins, atoms, symmetry, comm):
    kd = KPointDescriptor(bzkpts_kc, nspins)
    # self.timer.start('Set symmetry')
    kd.set_symmetry(atoms, symmetry, comm=comm)
    # self.timer.stop('Set symmetry')
    return kd


def create_mixed_kpoint_descriptor(bzk_coarse_kc, bzk_fine_kc, centers_i,
                                   weight_fine_k, kwargs):
    assert len(weight_fine_k) == bzk_fine_kc.shape[0]

    # Get old Monkhorst-Pack grid points and delete the centers from them
    nbzkpts_coarse = bzk_coarse_kc.shape[0]
    bzk_new_kc = np.delete(bzk_coarse_kc, centers_i, axis=0)
    nbzkpts_coarse_new = bzk_new_kc.shape[0]
    # Add refined points
    nbzkpts_fine = bzk_fine_kc.shape[0]
    bzk_new_kc = np.append(bzk_new_kc, bzk_fine_kc, axis=0)

    # Construct the new KPointDescriptor
    kd_new = create_kpoint_descriptor(bzk_new_kc, **kwargs)
    refine_info = KRefinement()
    refine_info.set_unrefined_nbzkpts(nbzkpts_coarse)
    label_k = np.array(nbzkpts_coarse_new * ['mh'] + nbzkpts_fine * ['refine'])
    refine_info.set_label_k(label_k)
    weight_k = np.append(np.ones(nbzkpts_coarse_new), weight_fine_k)
    refine_info.set_weight_k(weight_k)
    kd_new.refine_info = refine_info
    assert len(weight_k) == bzk_new_kc.shape[0]

    # This new Descriptor is good, except the weights are completely wrong now.
    kd_new.weight_k = np.bincount(kd_new.bz2ibz_k, weight_k) / nbzkpts_coarse
    assert np.abs(np.sum(kd_new.weight_k) - 1) < 1e-6

    # return kd_new.copy()
    return kd_new


def get_fine_bzkpts(center_ic, size, bzk_coarse_kc, kwargs):
    """Return list of reducible refined kpoints and the indexes for the points
    on the coarse kpoint grid. """

    # Define Monkhorst-Pack grid as starting point
    kd_coarse = create_kpoint_descriptor(bzk_coarse_kc, **kwargs)

    # Determine how many (cubic) nearest neighbour shells to replace
    nshells = size.shape[0]

    # Get coordinates of neighbour k-points. First index is the 'shell' aka
    # order of neighbour. Coordinates always centred around zero.
    neighbours_kc = construct_neighbours_by_shells(nshells, kd_coarse.N_c)

    # loop over all the different shells
    centers_i = np.empty((0), dtype=int)
    bzk_fine_kc = []
    weight_k = []
    for shell in range(nshells):
        # Determine all k-points in shell, which will be refinement centers
        shell_kc = []
        for i, center_c in enumerate(center_ic):
            shell_c = neighbours_kc[shell] + center_c
            shell_kc.append(shell_c)
        shell_kc = np.concatenate(shell_kc)

        # Find all equivalent centers using full symmetry
        center_i = []
        for k, shell_c in enumerate(shell_kc):
            equiv_i = find_equivalent_kpoints(shell_c, kd_coarse)
            center_i.append(equiv_i)
        center_i = np.concatenate(center_i)

        # Remove redundant entries, which come from symmetry
        center_i, index_map = np.unique(center_i, return_index=True)

        # Append to list for all shells
        # Remove also points which occur also in a previous shell. This can
        # happen due to symmetry or overlap of centers' shells
        center_i = np.setdiff1d(center_i, centers_i)
        centers_i = np.append(centers_i, center_i)

        # Assign the kpoint vectors corresponding to the indexes
        centers_kc = bzk_coarse_kc[center_i]

        # Get scaled Monkhorst-Pack for refinement
        mh_kc = get_reduced_monkhorst(size[shell], kd_coarse.N_c)

        # Gather the refined points of the grid for all centers
        for k, centers_c in enumerate(centers_kc):
            this_kc = mh_kc + centers_c
            bzk_fine_kc.append(this_kc)

        # Determine weight of refined points
        weight = 1. / np.prod(size[shell])
        nkpts = len(center_i) * np.prod(size[shell])
        weight *= np.ones(nkpts)
        weight_k.append(weight)

    weight_k = np.concatenate(weight_k)
    bzk_fine_kc = np.concatenate(bzk_fine_kc)

    assert np.abs(len(centers_i) - sum(weight_k)) < 1e-6
    assert len(weight_k) == bzk_fine_kc.shape[0]

    return centers_i, bzk_fine_kc, weight_k


def find_equivalent_kpoints(point, kd):
    """Find coordinates and BZ index for all refinement centers."""

    # check whether point in bzkpts_kc
    min_dist, k = minimal_point_distance(point, kd.bzk_kc)
    if min_dist > 1e-8:
        raise RuntimeError('Could not find k-point in kd list!')

    equiv_k = list(set(kd.bz2bz_ks[k]))
    # equiv_kc = kd.bzk_kc[equiv_k]

    return np.array(equiv_k)  # , equiv_kc


def get_reduced_monkhorst(size, N_c):
    """Find monkhorst_pack according to size of refined grid and shrink it into
    the volume of one original kpoint."""

    mh_kc = monkhorst_pack(size)
    return mh_kc / N_c


def construct_neighbours_by_shells(nshells, N_c):
    """Construct a list of neighbours (translations from center for each
    shell around the center."""

    neighbours = []

    # The innermost shell is always the center itself
    neighbours.append(np.array([0, 0, 0], dtype=float, ndmin=2))

    # Loop through the other shells
    for shell in range(1, nshells):
        # Construct the displacements/elements for each dimension.
        elements = []
        for i in range(3):
            if N_c[i] == 1:
                elements.append([0, ])
            else:
                elements.append(list(range(-shell, shell + 1)))

        # Construct the vectors
        # For each valid point, at least one component must have the value
        # of the shell index (+ or -).
        this_list = []
        for cx in elements[0]:
            for cy in elements[1]:
                for cz in elements[2]:
                    candidate = [cx, cy, cz]
                    if (np.abs(np.array(candidate)) == shell).any():
                        this_list.append(candidate)

        this_list = np.array(this_list, dtype=float, ndmin=2) / N_c
        neighbours.append(this_list)

    return np.array(neighbours)


def prune_symmetries_kpoints(kd, symmetry):
    """Remove the symmetries which are violated by a kpoint set."""

    new_symmetry = copy.copy(symmetry)

    nsyms = len(symmetry.op_scc)

    where = np.where(~np.any(kd.bz2bz_ks[:, 0:nsyms] == -1, 0))

    new_symmetry.op_scc = new_symmetry.op_scc[where]
    new_symmetry.ft_sc = new_symmetry.ft_sc[where]
    new_symmetry.a_sa = new_symmetry.a_sa[where]

    inv_cc = -np.eye(3, dtype=int)
    new_symmetry.has_inversion = (new_symmetry.op_scc ==
                                  inv_cc).all(2).all(1).any()

    return new_symmetry


def find_missing_points(kd):
    """Find points in k-point descriptor, which miss in the set to fill the
    group."""
    if -1 not in kd.bz2bz_ks:
        return None

    # Find unaccounted points
    buf = np.empty((0, 3))
    for k in range(kd.bz2bz_ks.shape[0]):
        for s in range(kd.bz2bz_ks.shape[1]):
            if kd.bz2bz_ks[k, s] == -1:
                sign = 1.0
                i = s
                if kd.symmetry.time_reversal:
                    if s / kd.bz2bz_ks.shape[1] >= 0.5:
                        i = s - int(kd.bz2bz_ks.shape[1] / 2)
                        sign = -1.
                k_c = sign * np.dot(kd.symmetry.op_scc[i], kd.bzk_kc[k])
                k_c -= np.rint(k_c)
                # Check that k_c hasn't been added yet
                if buf.shape[0] > 0:
                    min_dist, index = minimal_point_distance(k_c, buf)
                    if min_dist < 1e-6:
                        continue
                buf = np.concatenate((buf, np.array(k_c, ndmin=2)))

    return buf


def add_missing_points(kd, kwargs):
    """Add points to k-point descriptor, which miss to fill the group."""
    add_points_kc = find_missing_points(kd)
    if add_points_kc is None:
        return kd

    kd_new = create_new_descriptor_with_zero_points(kd, add_points_kc, kwargs)
    assert -1 not in kd_new.bz2bz_ks

    return kd_new


def add_plusq_points(kd, q_c, kwargs):
    """Add +q points to k-point descriptor, if missing. Also, reduce the
    symmetry of the system as necessary."""

    # Add missing points to retrieve full symmetry. Might be redundant.
    _kd = add_missing_points(kd, kwargs)

    # Find missing q
    add_points_kc = []
    for k in range(_kd.nbzkpts):
        # if q_c is small, use q_c = 0.0 for mh points, else they don't need
        # extra points anyway
        if _kd.refine_info.label_k[k] == 'mh':
            continue
        try:
            _kd.find_k_plus_q(q_c, [k])
        except KPointError:
            k_c = _kd.bzk_kc[k] + q_c
            add_points_kc.append(np.array(k_c, ndmin=2))
    add_points_kc = np.concatenate(add_points_kc)

    if add_points_kc.shape[0] == 0:
        return _kd

    # Find reduced +q symmetry
    bzk_kc = np.append(_kd.bzk_kc, add_points_kc, axis=0)
    _kd_new = create_kpoint_descriptor(bzk_kc, **kwargs)
    symm_new = prune_symmetries_kpoints(_kd_new, kwargs.get('symmetry'))
    kwargs['symmetry'] = symm_new
    del _kd_new, _kd

    kd_new = create_new_descriptor_with_zero_points(kd, add_points_kc, kwargs)
    return kd_new


def create_new_descriptor_with_zero_points(kd, add_points_kc, kwargs):
    """Create a new k-point descriptor including some additional zero-weighted
    points."""
    bzk_kc = np.append(kd.bzk_kc, add_points_kc, axis=0)
    nbzkpts_add = add_points_kc.shape[0]

    # Make sure all points are unique
    assert bzk_kc.shape[0] == np.vstack([tuple(row)
                                         for row in bzk_kc]).shape[0]

    # Construct the new KPointDescriptor
    kd_new = create_kpoint_descriptor(bzk_kc, **kwargs)

    # Update refine_info
    kd_new.refine_info = kd.refine_info.copy()
    label_k = np.append(kd.refine_info.label_k,
                        np.array(nbzkpts_add * ['zero']))
    kd_new.refine_info.set_label_k(label_k)
    # Avoid exact zero, as that would screw up the occupations calculation
    weight_k = np.append(kd.refine_info.weight_k, 1e-10 * np.ones(nbzkpts_add))
    kd_new.refine_info.set_weight_k(weight_k)

    # Correct ibz weights
    kd_new.weight_k = np.bincount(kd_new.bz2ibz_k, weight_k)
    kd_new.weight_k *= 1.0 / kd_new.refine_info.get_unrefined_nbzkpts()
    assert np.abs(np.sum(kd_new.weight_k) - 1) < 1e-6

    return kd_new


def minimal_point_distance(point_c, bzk_kc):
    d_kc = point_c - bzk_kc
    d_k = abs(d_kc - d_kc.round()).sum(1)
    k = d_k.argmin()
    return d_k[k], k


class KRefinement:
    """Additional information for refined kpoint grids.
    """
    def __init__(self):
        self.mhnbzkpts = None
        self.label_k = None
        self.weight_k = None
        self.almostoptical = None

    def __str__(self):
        s = "Using k-point grid refinement"
        if self.almostoptical:
            s += " with almostoptical approximation"
        s += "\n"
        return s

    def set_unrefined_nbzkpts(self, mhnbzkpts):
        self.mhnbzkpts = mhnbzkpts

    def get_unrefined_nbzkpts(self):
        return self.mhnbzkpts

    def set_weight_k(self, weight_k):
        self.weight_k = weight_k

    def get_weight_k(self):
        return self.weight_k

    def set_label_k(self, label_k):
        self.label_k = label_k

    def get_label_k(self):
        return self.label_k

    def copy(self):
        refine_info = KRefinement()
        refine_info.mhnbzkpts = self.mhnbzkpts
        refine_info.label_k = self.label_k
        refine_info.weight_k = self.weight_k
        refine_info.almostoptical = self.almostoptical
        return refine_info
