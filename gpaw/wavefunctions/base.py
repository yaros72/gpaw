import numpy as np
from ase.units import Hartree

from gpaw.projections import Projections
from gpaw.utilities import pack, unpack2
from gpaw.utilities.blas import gemm, axpy
from gpaw.utilities.partition import AtomPartition


class WaveFunctions:
    """...

    setups:
        List of setup objects.
    symmetry:
        Symmetry object.
    kpt_u:
        List of **k**-point objects.
    nbands: int
        Number of bands.
    nspins: int
        Number of spins.
    dtype: dtype
        Data type of wave functions (float or complex).
    bzk_kc: ndarray
        Scaled **k**-points used for sampling the whole
        Brillouin zone - values scaled to [-0.5, 0.5).
    ibzk_kc: ndarray
        Scaled **k**-points in the irreducible part of the
        Brillouin zone.
    weight_k: ndarray
        Weights of the **k**-points in the irreducible part
        of the Brillouin zone (summing up to 1).
    kpt_comm:
        MPI-communicator for parallelization over **k**-points.
    """

    def __init__(self, gd, nvalence, setups, bd, dtype, collinear,
                 world, kd, kptband_comm, timer):
        self.gd = gd
        self.nspins = kd.nspins
        self.collinear = collinear
        self.nvalence = nvalence
        self.bd = bd
        self.dtype = dtype
        assert dtype == float or dtype == complex
        self.world = world
        self.kd = kd
        self.kptband_comm = kptband_comm
        self.timer = timer
        self.atom_partition = None

        self.mykpts = kd.create_k_points(self.gd, collinear)

        self.eigensolver = None
        self.positions_set = False
        self.spos_ac = None

        self.set_setups(setups)

    @property
    def kpt_u(self):
        """Old name."""
        return self.mykpts

    def summary(self, log):
        log(eigenvalue_string(self))

    def set_setups(self, setups):
        self.setups = setups

    def set_eigensolver(self, eigensolver):
        self.eigensolver = eigensolver

    def add_realspace_orbital_to_density(self, nt_G, psit_G):
        if psit_G.dtype == float:
            axpy(1.0, psit_G**2, nt_G)
        else:
            assert psit_G.dtype == complex
            axpy(1.0, psit_G.real**2, nt_G)
            axpy(1.0, psit_G.imag**2, nt_G)

    def add_orbital_density(self, nt_G, kpt, n):
        self.add_realspace_orbital_to_density(nt_G, kpt.psit_nG[n])

    def calculate_density_contribution(self, nt_sG):
        """Calculate contribution to pseudo density from wave functions.

        Array entries are written to (not added to)."""
        nt_sG.fill(0.0)
        for kpt in self.kpt_u:
            self.add_to_density_from_k_point(nt_sG, kpt)
        self.kptband_comm.sum(nt_sG)

        self.timer.start('Symmetrize density')
        for nt_G in nt_sG:
            self.kd.symmetry.symmetrize(nt_G, self.gd)
        self.timer.stop('Symmetrize density')

    def add_to_density_from_k_point(self, nt_sG, kpt):
        self.add_to_density_from_k_point_with_occupation(nt_sG, kpt, kpt.f_n)

    def get_orbital_density_matrix(self, a, kpt, n):
        """Add the nth band density from kpt to density matrix D_sp"""
        ni = self.setups[a].ni
        D_sii = np.zeros((self.nspins, ni, ni))
        P_i = kpt.P_ani[a][n]
        D_sii[kpt.s] += np.outer(P_i.conj(), P_i).real
        D_sp = [pack(D_ii) for D_ii in D_sii]
        return D_sp

    def calculate_atomic_density_matrices_k_point(self, D_sii, kpt, a, f_n):
        if kpt.rho_MM is not None:
            P_Mi = self.P_aqMi[a][kpt.q]
            rhoP_Mi = np.zeros_like(P_Mi)
            D_ii = np.zeros(D_sii[kpt.s].shape, kpt.rho_MM.dtype)
            gemm(1.0, P_Mi, kpt.rho_MM, 0.0, rhoP_Mi)
            gemm(1.0, rhoP_Mi, P_Mi.T.conj().copy(), 0.0, D_ii)
            D_sii[kpt.s] += D_ii.real
        else:
            if self.collinear:
                P_ni = kpt.projections[a]
                D_sii[kpt.s] += np.dot(P_ni.T.conj() * f_n, P_ni).real
            else:
                P_nsi = kpt.projections[a]
                D_ssii = np.einsum('nsi,n,nzj->szij',
                                   P_nsi.conj(), f_n, P_nsi)
                D_sii[0] += (D_ssii[0, 0] + D_ssii[1, 1]).real
                D_sii[1] += 2 * D_ssii[0, 1].real
                D_sii[2] += 2 * D_ssii[0, 1].imag
                D_sii[3] += (D_ssii[0, 0] - D_ssii[1, 1]).real

        if hasattr(kpt, 'c_on'):
            for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                d_nn = ne * np.outer(c_n.conj(), c_n)
                D_sii[kpt.s] += np.dot(P_ni.T.conj(), np.dot(d_nn, P_ni)).real

    def calculate_atomic_density_matrices(self, D_asp):
        """Calculate atomic density matrices from projections."""
        f_un = [kpt.f_n for kpt in self.kpt_u]
        self.calculate_atomic_density_matrices_with_occupation(D_asp, f_un)

    def calculate_atomic_density_matrices_with_occupation(self, D_asp, f_un):
        """Calculate atomic density matrices from projections with
        custom occupation f_un."""

        # Parameter check (if user accidentally passes f_n instead of f_un)
        if f_un[0] is not None:  # special case for transport calculations...
            assert isinstance(f_un[0], np.ndarray)
        # Varying f_n used in calculation of response part of GLLB-potential
        for a, D_sp in D_asp.items():
            ni = self.setups[a].ni
            D_sii = np.zeros((len(D_sp), ni, ni))
            for f_n, kpt in zip(f_un, self.kpt_u):
                self.calculate_atomic_density_matrices_k_point(D_sii, kpt, a,
                                                               f_n)
            D_sp[:] = [pack(D_ii) for D_ii in D_sii]
            self.kptband_comm.sum(D_sp)

        self.symmetrize_atomic_density_matrices(D_asp)

    def symmetrize_atomic_density_matrices(self, D_asp):
        if len(self.kd.symmetry.op_scc) == 0:
            return

        a_sa = self.kd.symmetry.a_sa
        D_asp.redistribute(self.atom_partition.as_serial())
        for s in range(self.nspins):
            D_aii = [unpack2(D_asp[a][s])
                     for a in range(len(D_asp))]
            for a, D_ii in enumerate(D_aii):
                setup = self.setups[a]
                D_asp[a][s] = pack(setup.symmetrize(a, D_aii, a_sa))
        D_asp.redistribute(self.atom_partition)

    def set_positions(self, spos_ac, atom_partition=None):
        self.positions_set = False
        # rank_a = self.gd.get_ranks_from_positions(spos_ac)
        # atom_partition = AtomPartition(self.gd.comm, rank_a)
        # XXX pass AtomPartition around instead of spos_ac?
        # All the classes passing around spos_ac end up needing the ranks
        # anyway.

        if atom_partition is None:
            rank_a = self.gd.get_ranks_from_positions(spos_ac)
            atom_partition = AtomPartition(self.gd.comm, rank_a)

        if self.atom_partition is not None and self.kpt_u[0].P_ani is not None:
            with self.timer('Redistribute'):
                for kpt in self.mykpts:
                    P = kpt.projections
                    assert self.atom_partition == P.atom_partition
                    kpt.projections = P.redist(atom_partition)
                    assert atom_partition == kpt.projections.atom_partition

        self.atom_partition = atom_partition
        self.kd.symmetry.check(spos_ac)
        self.spos_ac = spos_ac

    def allocate_arrays_for_projections(self, my_atom_indices):  # XXX unused
        if not self.positions_set and self.mykpts[0].projections is not None:
            # Projections have been read from file - don't delete them!
            pass
        else:
            nproj_a = [setup.ni for setup in self.setups]
            for kpt in self.mykpts:
                kpt.projections = Projections(
                    self.bd.nbands, nproj_a,
                    self.atom_partition,
                    self.bd.comm,
                    collinear=self.collinear, spin=kpt.s, dtype=self.dtype)

    def collect_eigenvalues(self, k, s):
        return self.collect_array('eps_n', k, s)

    def collect_occupations(self, k, s):
        return self.collect_array('f_n', k, s)

    def collect_array(self, name, k, s, subset=None):
        """Helper method for collect_eigenvalues and collect_occupations.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_u = self.kpt_u
        kpt_rank, u = self.kd.get_rank_and_index(s, k)
        if self.kd.comm.rank == kpt_rank:
            a_nx = getattr(kpt_u[u], name)

            if subset is not None:
                a_nx = a_nx[subset]

            # Domain master send this to the global master
            if self.gd.comm.rank == 0:
                if self.bd.comm.size == 1:
                    if kpt_rank == 0:
                        return a_nx
                    else:
                        self.kd.comm.ssend(a_nx, 0, 1301)
                else:
                    b_nx = self.bd.collect(a_nx)
                    if self.bd.comm.rank == 0:
                        if kpt_rank == 0:
                            return b_nx
                        else:
                            self.kd.comm.ssend(b_nx, 0, 1301)

        elif self.world.rank == 0 and kpt_rank != 0:
            # Only used to determine shape and dtype of receiving buffer:
            a_nx = getattr(kpt_u[0], name)

            if subset is not None:
                a_nx = a_nx[subset]

            b_nx = np.zeros((self.bd.nbands,) + a_nx.shape[1:],
                            dtype=a_nx.dtype)
            self.kd.comm.receive(b_nx, kpt_rank, 1301)
            return b_nx

        return np.zeros(0)  # see comment in get_wave_function_array() method

    def collect_auxiliary(self, value, k, s, shape=1, dtype=float):
        """Helper method for collecting band-independent scalars/arrays.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_u = self.kpt_u
        kpt_rank, u = self.kd.get_rank_and_index(s, k)

        if self.kd.comm.rank == kpt_rank:
            if isinstance(value, str):
                a_o = getattr(kpt_u[u], value)
            else:
                a_o = value[u]  # assumed list

            # Make sure data is a mutable object
            a_o = np.asarray(a_o)

            if a_o.dtype is not dtype:
                a_o = a_o.astype(dtype)

            # Domain master send this to the global master
            if self.gd.comm.rank == 0:
                if kpt_rank == 0:
                    return a_o
                else:
                    self.kd.comm.send(a_o, 0, 1302)

        elif self.world.rank == 0 and kpt_rank != 0:
            b_o = np.zeros(shape, dtype=dtype)
            self.kd.comm.receive(b_o, kpt_rank, 1302)
            return b_o

    def collect_projections(self, k, s):
        """Helper method for collecting projector overlaps across domains.

        For the parallel case find the rank in kpt_comm that contains
        the (k,s) pair, for this rank, send to the global master."""

        kpt_rank, u = self.kd.get_rank_and_index(s, k)

        if self.kd.comm.rank == kpt_rank:
            kpt = self.mykpts[u]
            P_nI = kpt.projections.collect()
            if self.world.rank == 0:
                return P_nI
            if P_nI is not None:
                self.kd.comm.send(np.ascontiguousarray(P_nI), 0)
        if self.world.rank == 0:
            nproj = sum(setup.ni for setup in self.setups)
            if not self.collinear:
                nproj *= 2
            P_nI = np.empty((self.bd.nbands, nproj), self.dtype)
            self.kd.comm.receive(P_nI, kpt_rank)
            return P_nI

    def get_wave_function_array(self, n, k, s, realspace=True, periodic=False):
        """Return pseudo-wave-function array on master.

        n: int
            Global band index.
        k: int
            Global IBZ k-point index.
        s: int
            Spin index (0 or 1).
        realspace: bool
            Transform plane wave or LCAO expansion coefficients to real-space.

        For the parallel case find the ranks in kd.comm and bd.comm
        that contains to (n, k, s), and collect on the corresponding
        domain a full array on the domain master and send this to the
        global master."""

        kpt_rank, u = self.kd.get_rank_and_index(s, k)
        band_rank, myn = self.bd.who_has(n)

        rank = self.world.rank

        if (self.kd.comm.rank == kpt_rank and
            self.bd.comm.rank == band_rank):
            psit_G = self._get_wave_function_array(u, myn, realspace, periodic)

            if realspace:
                psit_G = self.gd.collect(psit_G)

            if rank == 0:
                return psit_G

            # Domain master send this to the global master
            if self.gd.comm.rank == 0:
                self.world.ssend(psit_G, 0, 1398)

        if rank == 0:
            # allocate full wave function and receive
            shape = () if self.collinear else (2,)
            psit_G = self.empty(shape, global_array=True,
                                realspace=realspace)
            # XXX this will fail when using non-standard nesting
            # of communicators.
            world_rank = (kpt_rank * self.gd.comm.size *
                          self.bd.comm.size +
                          band_rank * self.gd.comm.size)
            self.world.receive(psit_G, world_rank, 1398)
            return psit_G

        # We return a number instead of None on all the slaves.  Most of
        # the time the return value will be ignored on the slaves, but
        # in some cases it will be multiplied by some other number and
        # then ignored.  Allowing for this will simplify some code here
        # and there.
        return np.nan

    def get_homo_lumo(self, spin=None):
        """Return HOMO and LUMO eigenvalues."""
        if spin is None:
            if self.nspins == 1:
                return self.get_homo_lumo(0)
            h0, l0 = self.get_homo_lumo(0)
            h1, l1 = self.get_homo_lumo(1)
            return np.array([max(h0, h1), min(l0, l1)])

        n = self.nvalence // 2
        band_rank, myn = self.bd.who_has(n - 1)
        homo = -np.inf
        if self.bd.comm.rank == band_rank:
            for kpt in self.kpt_u:
                if kpt.s == spin:
                    homo = max(kpt.eps_n[myn], homo)
        homo = self.world.max(homo)

        lumo = np.inf
        if n < self.bd.nbands:  # there are not enough bands for LUMO
            band_rank, myn = self.bd.who_has(n)
            if self.bd.comm.rank == band_rank:
                for kpt in self.kpt_u:
                    if kpt.s == spin:
                        lumo = min(kpt.eps_n[myn], lumo)
            lumo = self.world.min(lumo)

        return np.array([homo, lumo])

    def write(self, writer):
        writer.write(version=1, ha=Hartree)
        writer.write(kpts=self.kd)
        self.write_projections(writer)
        self.write_eigenvalues(writer)
        self.write_occupations(writer)

    def write_projections(self, writer):
        nproj = sum(setup.ni for setup in self.setups)

        if self.collinear:
            shape = (self.nspins, self.kd.nibzkpts, self.bd.nbands, nproj)
        else:
            shape = (self.kd.nibzkpts, self.bd.nbands, 2, nproj)

        writer.add_array('projections', shape, self.dtype)

        for s in range(self.nspins):
            for k in range(self.kd.nibzkpts):
                P_nI = self.collect_projections(k, s)
                if not self.collinear and P_nI is not None:
                    P_nI.shape = (self.bd.nbands, 2, nproj)
                writer.fill(P_nI)

    def write_eigenvalues(self, writer):
        if self.collinear:
            shape = (self.nspins, self.kd.nibzkpts, self.bd.nbands)
        else:
            shape = (self.kd.nibzkpts, self.bd.nbands)

        writer.add_array('eigenvalues', shape)
        for s in range(self.nspins):
            for k in range(self.kd.nibzkpts):
                writer.fill(self.collect_eigenvalues(k, s) * Hartree)

    def write_occupations(self, writer):

        if self.collinear:
            shape = (self.nspins, self.kd.nibzkpts, self.bd.nbands)
        else:
            shape = (self.kd.nibzkpts, self.bd.nbands)

        writer.add_array('occupations', shape)
        for s in range(self.nspins):
            for k in range(self.kd.nibzkpts):
                # Scale occupation numbers when writing:
                # XXX fix this in the code also ...
                weight = self.kd.weight_k[k] * 2 / self.nspins
                writer.fill(self.collect_occupations(k, s) / weight)

    def read(self, reader):
        r = reader.wave_functions
        # Backward compatibility:
        # Take parameters from main reader
        if 'ha' not in r:
            r.ha = reader.ha
        if 'version' not in r:
            r.version = reader.version
        self.read_projections(r)
        self.read_eigenvalues(r, r.version == 0)
        self.read_occupations(r, r.version == 0)

    def read_projections(self, reader):
        nslice = self.bd.get_slice()
        nproj_a = [setup.ni for setup in self.setups]
        atom_partition = AtomPartition(self.gd.comm,
                                       np.zeros(len(nproj_a), int))
        for u, kpt in enumerate(self.kpt_u):
            if self.collinear:
                index = (kpt.s, kpt.k)
            else:
                index = (kpt.k,)
            kpt.projections = Projections(
                self.bd.nbands, nproj_a,
                atom_partition, self.bd.comm,
                collinear=self.collinear, spin=kpt.s, dtype=self.dtype)
            if self.gd.comm.rank == 0:
                P_nI = reader.proxy('projections', *index)[nslice]
                if not self.collinear:
                    P_nI.shape = (self.bd.mynbands, -1)
                kpt.projections.matrix.array[:] = P_nI

    def read_eigenvalues(self, reader, old=False):
        nslice = self.bd.get_slice()
        for u, kpt in enumerate(self.kpt_u):
            if self.collinear:
                index = (kpt.s, kpt.k)
            else:
                index = (kpt.k,)
            eps_n = reader.proxy('eigenvalues', *index)[nslice]
            x = self.bd.mynbands - len(eps_n)  # missing bands?
            if x > 0:
                # Working on a real fix to this parallelization problem ...
                eps_n = np.pad(eps_n, (0, x), 'constant')
            if not old:  # skip for old tar-files gpw's
                eps_n /= reader.ha
            kpt.eps_n = eps_n

    def read_occupations(self, reader, old=False):
        nslice = self.bd.get_slice()
        for u, kpt in enumerate(self.kpt_u):
            if self.collinear:
                index = (kpt.s, kpt.k)
            else:
                index = (kpt.k,)
            f_n = reader.proxy('occupations', *index)[nslice]
            x = self.bd.mynbands - len(f_n)  # missing bands?
            if x > 0:
                # Working on a real fix to this parallelization problem ...
                f_n = np.pad(f_n, (0, x), 'constant')
            if not old:  # skip for old tar-files gpw's
                f_n *= kpt.weight
            kpt.f_n = f_n


def eigenvalue_string(wfs, comment=' '):
    """Write eigenvalues and occupation numbers into a string.

    The parameter comment can be used to comment out non-numers,
    for example to escape it for gnuplot.
    """

    tokens = []

    def add(*line):
        for token in line:
            tokens.append(token)
        tokens.append('\n')

    def eigs(k, s):
        eps_n = wfs.collect_eigenvalues(k, s)
        return eps_n * Hartree

    occs = wfs.collect_occupations

    if len(wfs.kd.ibzk_kc) == 1:
        if wfs.nspins == 1:
            add(comment, 'Band  Eigenvalues  Occupancy')
            eps_n = eigs(0, 0)
            f_n = occs(0, 0)
            if wfs.world.rank == 0:
                for n in range(wfs.bd.nbands):
                    add('%5d  %11.5f  %9.5f' % (n, eps_n[n], f_n[n]))
        else:
            add(comment, '                  Up                     Down')
            add(comment, 'Band  Eigenvalues  Occupancy  Eigenvalues  '
                'Occupancy')
            epsa_n = eigs(0, 0)
            epsb_n = eigs(0, 1)
            fa_n = occs(0, 0)
            fb_n = occs(0, 1)
            if wfs.world.rank == 0:
                for n in range(wfs.bd.nbands):
                    add('%5d  %11.5f  %9.5f  %11.5f  %9.5f' %
                        (n, epsa_n[n], fa_n[n], epsb_n[n], fb_n[n]))
        return ''.join(tokens)

    if len(wfs.kd.ibzk_kc) > 2:
        add('Showing only first 2 kpts')
        print_range = 2
    else:
        add('Showing all kpts')
        print_range = len(wfs.kd.ibzk_kc)

    if wfs.nvalence / 2. > 2:
        m = int(wfs.nvalence / 2. - 2)
    else:
        m = 0
    if wfs.bd.nbands - wfs.nvalence / 2. > 2:
        j = int(wfs.nvalence / 2. + 2)
    else:
        j = int(wfs.bd.nbands)

    if wfs.nspins == 1:
        add(comment, 'Kpt  Band  Eigenvalues  Occupancy')
        for i in range(print_range):
            eps_n = eigs(i, 0)
            f_n = occs(i, 0)
            if wfs.world.rank == 0:
                for n in range(m, j):
                    add('%3i %5d  %11.5f  %9.5f' % (i, n, eps_n[n], f_n[n]))
                add()
    else:
        add(comment, '                     Up                     Down')
        add(comment, 'Kpt  Band  Eigenvalues  Occupancy  Eigenvalues  '
            'Occupancy')

        for i in range(print_range):
            epsa_n = eigs(i, 0)
            epsb_n = eigs(i, 1)
            fa_n = occs(i, 0)
            fb_n = occs(i, 1)
            if wfs.world.rank == 0:
                for n in range(m, j):
                    add('%3i %5d  %11.5f  %9.5f  %11.5f  %9.5f' %
                        (i, n, epsa_n[n], fa_n[n], epsb_n[n], fb_n[n]))
                add()
    return ''.join(tokens)
