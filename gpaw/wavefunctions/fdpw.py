from __future__ import division
import numpy as np
from ase.utils.timing import timer

from gpaw.lcao.eigensolver import DirectLCAO
from gpaw.lfc import BasisFunctions
from gpaw.matrix import Matrix, matrix_matrix_multiply as mmm
from gpaw.utilities import unpack
from gpaw.utilities.timing import nulltimer
from gpaw.wavefunctions.base import WaveFunctions
from gpaw.wavefunctions.lcao import LCAOWaveFunctions, update_phases


class NullWfsMover:
    description = 'Wavefunctions kept unchanged if atoms move'

    def initialize(self, lcaowfs):
        pass

    def cut_wfs(self, wfs, spos_ac):
        pass


class PseudoPartialWaveWfsMover:
    """Move wavefunctions with atoms according to PAW basis

    Wavefunctions are approximated around atom a as

       ~          --  ~ a      ~a    ~
      psi (r)  ~  >  phi (r) < p  | psi >
         n        --    i       i      n
                  ai

    This quantity is then subtracted and re-added at the new
    positions.
    """
    description = 'Improved wavefunction reuse through dual PAW basis'

    def initialize(self, lcaowfs):
        pass

    def cut_wfs(self, wfs, spos_ac):
        ni_a = {}

        for a in range(len(wfs.setups)):
            setup = wfs.setups[a]
            l_j = [phit.get_angular_momentum_number()
                   for phit in setup.get_partial_waves_for_atomic_orbitals()]
            #assert l_j == setup.l_j[:len(l_j)]  # Relationship to l_orb_j?
            ni_a[a] = sum(2 * l + 1 for l in l_j)

        phit = wfs.get_pseudo_partial_waves()
        phit.set_positions(wfs.spos_ac, wfs.atom_partition)

        # XXX See also wavefunctions.lcao.update_phases
        if wfs.dtype == complex:
            phase_qa = np.exp(2j * np.pi *
                              np.dot(wfs.kd.ibzk_qc,
                                     (spos_ac - wfs.spos_ac).T.round()))

        def add_phit_to_wfs(multiplier):
            for kpt in wfs.kpt_u:
                P_ani = {}
                for a in kpt.P_ani:
                    P_ani[a] = multiplier * kpt.P_ani[a][:, :ni_a[a]]
                    if multiplier > 0 and wfs.dtype == complex:
                        P_ani[a] *= phase_qa[kpt.q, a]
                phit.add(kpt.psit_nG, c_axi=P_ani, q=kpt.q)

        add_phit_to_wfs(-1.0)

        def paste():
            phit.set_positions(spos_ac, wfs.atom_partition)
            add_phit_to_wfs(1.0)

        return paste


class LCAOWfsMover:
    """Move wavefunctions with atoms according to LCAO basis.

    Approximate wavefunctions as a linear combination of atomic
    orbitals, then subtract that linear combination and re-add
    it after moving the atoms using the same coefficients.

    The coefficients c are determined by the equation

               /    *  _  ^  ~   _   _    --
      X     =  | Phi  (r) S psi (r) dr =  >  S      c
       n mu    /    mu         n          --  mu nu  nu n
                                          nu

    We calculate X directly and then solve for c.
    """
    description = 'Improved wavefunction reuse through full LCAO basis'

    # TODO/FIXME
    # * Get rid of the unnecessary T matrix
    # * Only recalculate S/P when necessary (not first time)
    # * Full parallelization support (ScaLAPACK, check scipy atomic correction)
    #   Also replace np.linalg.solve by parallel/efficient thing
    # * Broken with PW mode because PW mode has very funny P_ani shapes.
    #   Also PW does not use the overlap object; this may be related
    # * Can we use updated S matrix to construct better guess?

    def initialize(self, lcaowfs):
        self.bfs = lcaowfs.basis_functions
        # self.tci = lcaowfs.tci
        self.tciexpansions = lcaowfs.tciexpansions
        self.atomic_correction = lcaowfs.atomic_correction
        # self.S_qMM = lcaowfs.S_qMM
        # self.T_qMM = lcaowfs.T_qMM  # Get rid of this
        # self.P_aqMi = lcaowfs.P_aqMi

    def cut_wfs(self, wfs, spos_ac):
        # XXX Must forward vars from LCAO initialization object
        # in order to not need to recalculate them.
        # Also, if we get the vars from the LCAO init object,
        # we can rely on those parallelization settings without danger.
        bfs = self.bfs

        # P_aqMi = self.P_aqMi
        # S_qMM = self.S_qMM

        # We can inherit S_qMM and P_aqMi from the initialization in the
        # first step, then recalculate them for subsequent steps.
        wfs.timer.start('reuse wfs')
        wfs.timer.start('tci calculate')
        tciex = self.tciexpansions
        manytci = tciex.get_manytci_calculator(wfs.setups, wfs.gd,
                                               spos_ac, wfs.kd.ibzk_qc,
                                               wfs.dtype, wfs.timer)
        P_aqMi = manytci.P_aqMi(bfs.my_atom_indices)
        # Avoid calculating T
        Mstart, Mstop = wfs.initksl.Mstart, wfs.initksl.Mstop
        S_qMM, T_qMM = manytci.O_qMM_T_qMM(wfs.gd.comm, Mstart, Mstop)
        wfs.timer.stop('tci calculate')
        self.atomic_correction.initialize(P_aqMi, Mstart, Mstop)
        # self.atomic_correction.gobble_data(wfs)
        wfs.timer.start('lcao overlap correction')
        self.atomic_correction.add_overlap_correction(wfs, S_qMM)
        wfs.timer.stop('lcao overlap correction')
        wfs.gd.comm.sum(S_qMM)
        c_unM = []
        for kpt in wfs.kpt_u:
            S_MM = S_qMM[kpt.q]
            X_nM = np.zeros((wfs.bd.mynbands, wfs.setups.nao), wfs.dtype)
            # XXX use some blocksize to reduce memory usage?
            with wfs.timer('wfs overlap'):
                opsit = kpt.psit.new()
                opsit.array[:] = kpt.psit.array
                opsit.add(wfs.pt, wfs.setups.dS.apply(kpt.projections))

            with wfs.timer('bfs integrate'):
                bfs.integrate2(opsit.array, c_xM=X_nM, q=kpt.q)

            with wfs.timer('gd comm sum'):
                wfs.gd.comm.sum(X_nM)

            # Mind band parallelization / ScaLAPACK
            # Actually we can probably ignore ScaLAPACK for FD/PW calculations
            # since we never adapted Davidson to those.  Although people
            # may have requested ScaLAPACK for LCAO initialization.
            c_nM = np.linalg.solve(S_MM.T, X_nM.T).T.copy()

            # c_nM *= 0  # This disables the whole mechanism
            with wfs.timer('lcao to grid'):
                bfs.lcao_to_grid(C_xM=-c_nM, psit_xG=kpt.psit_nG, q=kpt.q)

            c_unM.append(c_nM)

        if wfs.dtype == complex:
            update_phases(c_unM, [kpt.q for kpt in wfs.kpt_u], wfs.kd.ibzk_qc,
                          spos_ac, wfs.spos_ac, wfs.setups, wfs.initksl.Mstart)

        del opsit

        with wfs.timer('bfs set pos'):
            bfs.set_positions(spos_ac)

        # Is it possible to recalculate the overlaps and make use of how
        # they have changed here?
        wfs.timer.start('re-add wfs')
        for u, kpt in enumerate(wfs.kpt_u):
            bfs.lcao_to_grid(C_xM=c_unM[u], psit_xG=kpt.psit_nG, q=kpt.q)
        wfs.timer.stop('re-add wfs')
        wfs.timer.stop('reuse wfs')


class FDPWWaveFunctions(WaveFunctions):
    """Base class for finite-difference and planewave classes."""
    def __init__(self, parallel, initksl, reuse_wfs_method=None, **kwargs):
        WaveFunctions.__init__(self, **kwargs)

        self.scalapack_parameters = parallel

        self.initksl = initksl
        if reuse_wfs_method is None or reuse_wfs_method == 'keep':
            wfs_mover = NullWfsMover()
        elif hasattr(reuse_wfs_method, 'cut_wfs'):
            wfs_mover = reuse_wfs_method
        elif reuse_wfs_method == 'paw':
            wfs_mover = PseudoPartialWaveWfsMover()
        elif reuse_wfs_method == 'lcao':
            wfs_mover = LCAOWfsMover()
        else:
            raise ValueError('Strange way to reuse wfs: {}'
                             .format(reuse_wfs_method))

        self.wfs_mover = wfs_mover

        self.set_orthonormalized(False)

        self._work_matrix_nn = None  # storage for H, S, ...
        self._work_array = None

    @property
    def work_array(self):
        if self._work_array is None:
            self._work_array = self.empty(self.bd.mynbands *
                                          (1 if self.collinear else 2))
        return self._work_array

    @property
    def work_matrix_nn(self):
        """Get Matrix object for H, S, ..."""
        if self._work_matrix_nn is None:
            self._work_matrix_nn = Matrix(
                self.bd.nbands, self.bd.nbands,
                self.dtype,
                dist=(self.bd.comm, self.bd.comm.size))
        return self._work_matrix_nn

    def __str__(self):
        comm, r, c, b = self.scalapack_parameters
        L1 = ('  ScaLapack parameters: grid={}x{}, blocksize={}'
              .format(r, c, b))
        L2 = ('  Wavefunction extrapolation:\n    {}'
              .format(self.wfs_mover.description))
        return '\n'.join([L1, L2])

    def set_setups(self, setups):
        WaveFunctions.set_setups(self, setups)

    def set_orthonormalized(self, flag):
        self.orthonormalized = flag

    def set_positions(self, spos_ac, atom_partition=None):
        move_wfs = (self.kpt_u[0].psit_nG is not None and
                    self.spos_ac is not None)

        if move_wfs:
            paste_wfs = self.wfs_mover.cut_wfs(self, spos_ac)

        # This will update the positions -- and transfer, if necessary --
        # the projection matrices which may be necessary for updating
        # the wavefunctions.
        WaveFunctions.set_positions(self, spos_ac, atom_partition)

        if move_wfs and paste_wfs is not None:
            paste_wfs()

        self.set_orthonormalized(False)
        self.pt.set_positions(spos_ac, atom_partition)
        self.allocate_arrays_for_projections(self.pt.my_atom_indices)
        self.positions_set = True

    def initialize(self, density, hamiltonian, spos_ac):
        """Initialize wave-functions, density and hamiltonian.

        Return (nlcao, nrand) tuple with number of bands intialized from
        LCAO and random numbers, respectively."""

        if self.mykpts[0].psit is None:
            basis_functions = BasisFunctions(self.gd,
                                             [setup.phit_j
                                              for setup in self.setups],
                                             self.kd, dtype=self.dtype,
                                             cut=True)
            basis_functions.set_positions(spos_ac)
        else:
            self.initialize_wave_functions_from_restart_file()

        if self.mykpts[0].psit is not None:
            density.initialize_from_wavefunctions(self)
        elif density.nt_sG is None:
            density.initialize_from_atomic_densities(basis_functions)
            # Initialize GLLB-potential from basis function orbitals
            if hamiltonian.xc.type == 'GLLB':
                hamiltonian.xc.initialize_from_atomic_orbitals(
                    basis_functions)
        else:  # XXX???
            # We didn't even touch density, but some combinations in paw.set()
            # will make it necessary to do this for some reason.
            density.calculate_normalized_charges_and_mix()
        hamiltonian.update(density)

        if self.mykpts[0].psit is None:
            if 1:  # self.collinear:
                nlcao = self.initialize_wave_functions_from_basis_functions(
                    basis_functions, density, hamiltonian, spos_ac)
            else:
                self.random_wave_functions(0)
                nlcao = 0
            nrand = self.bd.nbands - nlcao
        else:
            # We got everything from file:
            nlcao = 0
            nrand = 0

        return nlcao, nrand

    def initialize_wave_functions_from_restart_file(self):
        for kpt in self.mykpts:
            if not kpt.psit.in_memory:
                kpt.psit.read_from_file()

    def initialize_wave_functions_from_basis_functions(self,
                                                       basis_functions,
                                                       density, hamiltonian,
                                                       spos_ac):
        # if self.initksl is None:
        #     raise RuntimeError('use fewer bands or more basis functions')

        self.timer.start('LCAO initialization')
        lcaoksl, lcaobd = self.initksl, self.initksl.bd
        lcaowfs = LCAOWaveFunctions(lcaoksl, self.gd, self.nvalence,
                                    self.setups, lcaobd, self.dtype,
                                    self.world, self.kd, self.kptband_comm,
                                    nulltimer)
        lcaowfs.basis_functions = basis_functions
        lcaowfs.timer = self.timer
        self.timer.start('Set positions (LCAO WFS)')
        lcaowfs.set_positions(spos_ac, self.atom_partition)
        self.timer.stop('Set positions (LCAO WFS)')

        if self.collinear:
            eigensolver = DirectLCAO()
        else:
            from gpaw.xc.noncollinear import NonCollinearLCAOEigensolver
            eigensolver = NonCollinearLCAOEigensolver()

        eigensolver.initialize(self.gd, self.dtype, self.setups.nao, lcaoksl)

        # XXX when density matrix is properly distributed, be sure to
        # update the density here also
        eigensolver.iterate(hamiltonian, lcaowfs)

        # Transfer coefficients ...
        for kpt, lcaokpt in zip(self.kpt_u, lcaowfs.kpt_u):
            kpt.C_nM = lcaokpt.C_nM

        self.wfs_mover.initialize(lcaowfs)

        # and get rid of potentially big arrays early:
        del eigensolver, lcaowfs

        with self.timer('LCAO to grid'):
            self.initialize_from_lcao_coefficients(basis_functions)

        if self.collinear and self.bd.mynbands > lcaobd.mynbands:
            # Add extra states.  If the number of atomic orbitals is
            # less than the desired number of bands, then extra random
            # wave functions are added.
            self.random_wave_functions(lcaobd.mynbands)
            # IMPORTANT: This intersperses random wavefunctions
            # with those from LCAO depending on band parallelization.
            # This is presumably okay as long as the FD/PW eigensolver
            # is called again before using the wavefunctions/occupations.
            #
            # Indeed as of writing this, the initialization appears to
            # call these things in the correct order, but there is no
            # telling when this will break due to some unrelated change.
        self.timer.stop('LCAO initialization')

        return lcaobd.nbands

    @timer('Orthonormalize')
    def orthonormalize(self, kpt=None):
        if kpt is None:
            for kpt in self.mykpts:
                self.orthonormalize(kpt)
            self.orthonormalized = True
            return

        psit = kpt.psit
        P = kpt.projections

        with self.timer('projections'):
            psit.matrix_elements(self.pt, out=P)

        S = self.work_matrix_nn
        P2 = P.new()

        with self.timer('calc_s_matrix'):
            psit.matrix_elements(out=S, symmetric=True, cc=True)
            self.setups.dS.apply(P, out=P2)
            mmm(1.0, P, 'N', P2, 'C', 1.0, S, symmetric=True)

        with self.timer('inverse-cholesky'):
            S.invcholesky()
            # S now contains the inverse of the Cholesky factorization

        psit2 = psit.new(buf=self.work_array)
        with self.timer('rotate_psi_s'):
            mmm(1.0, S, 'N', psit, 'N', 0.0, psit2)
            mmm(1.0, S, 'N', P, 'N', 0.0, P2)
            psit[:] = psit2
            kpt.projections = P2

    def calculate_forces(self, hamiltonian, F_av):
        # Calculate force-contribution from k-points:
        F_av[:] = 0.0

        if not self.collinear:
            F_ansiv = self.pt.dict(2 * self.bd.mynbands, derivative=True)
            dH_axp = hamiltonian.dH_asp
            for kpt in self.kpt_u:
                array = kpt.psit.array
                self.pt.derivative(array.reshape((-1, array.shape[-1])),
                                   F_ansiv, kpt.q)
                for a, F_nsiv in F_ansiv.items():
                    F_nsiv = F_nsiv.reshape((self.bd.mynbands,
                                             2, -1, 3)).conj()
                    F_nsiv *= kpt.f_n[:, np.newaxis, np.newaxis, np.newaxis]
                    dH_ii = unpack(dH_axp[a][0])
                    dH_vii = [unpack(dH_p) for dH_p in dH_axp[a][1:]]
                    dH_ssii = np.array(
                        [[dH_ii + dH_vii[2], dH_vii[0] - 1j * dH_vii[1]],
                         [dH_vii[0] + 1j * dH_vii[1], dH_ii - dH_vii[2]]])
                    P_nsi = kpt.projections[a]
                    F_v = np.einsum('nsiv,stij,ntj', F_nsiv, dH_ssii, P_nsi)
                    F_nsiv *= kpt.eps_n[:, np.newaxis, np.newaxis, np.newaxis]
                    dO_ii = self.setups[a].dO_ii
                    F_v -= np.einsum('nsiv,ij,nsj', F_nsiv, dO_ii, P_nsi)
                    F_av[a] += 2 * F_v.real

            self.bd.comm.sum(F_av, 0)

            if self.bd.comm.rank == 0:
                self.kd.comm.sum(F_av, 0)
            return

        F_aniv = self.pt.dict(self.bd.mynbands, derivative=True)
        dH_asp = hamiltonian.dH_asp
        for kpt in self.kpt_u:
            self.pt.derivative(kpt.psit_nG, F_aniv, kpt.q)
            for a, F_niv in F_aniv.items():
                F_niv = F_niv.conj()
                F_niv *= kpt.f_n[:, np.newaxis, np.newaxis]
                dH_ii = unpack(dH_asp[a][kpt.s])
                P_ni = kpt.P_ani[a]
                F_vii = np.dot(np.dot(F_niv.transpose(), P_ni), dH_ii)
                F_niv *= kpt.eps_n[:, np.newaxis, np.newaxis]
                dO_ii = self.setups[a].dO_ii
                F_vii -= np.dot(np.dot(F_niv.transpose(), P_ni), dO_ii)
                F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

            # Hack used in delta-scf calculations:
            if hasattr(kpt, 'c_on'):
                assert self.bd.comm.size == 1
                self.pt.derivative(kpt.psit_nG, F_aniv, kpt.q)  # XXX again
                d_nn = np.zeros((self.bd.mynbands, self.bd.mynbands),
                                dtype=complex)
                for ne, c_n in zip(kpt.ne_o, kpt.c_on):
                    d_nn += ne * np.outer(c_n.conj(), c_n)
                for a, F_niv in F_aniv.items():
                    F_niv = F_niv.conj()
                    dH_ii = unpack(dH_asp[a][kpt.s])
                    Q_ni = np.dot(d_nn, kpt.P_ani[a])
                    F_vii = np.dot(np.dot(F_niv.transpose(), Q_ni), dH_ii)
                    F_niv *= kpt.eps_n[:, np.newaxis, np.newaxis]
                    dO_ii = self.setups[a].dO_ii
                    F_vii -= np.dot(np.dot(F_niv.transpose(), Q_ni), dO_ii)
                    F_av[a] += 2 * F_vii.real.trace(0, 1, 2)

        self.bd.comm.sum(F_av, 0)

        if self.bd.comm.rank == 0:
            self.kd.comm.sum(F_av, 0)

    def estimate_memory(self, mem):
        gridbytes = self.bytes_per_wave_function()
        n = len(self.kpt_u) * self.bd.mynbands
        mem.subnode('Arrays psit_nG', n * gridbytes)
        self.eigensolver.estimate_memory(mem.subnode('Eigensolver'), self)
        ni = sum(dataset.ni for dataset in self.setups) / self.gd.comm.size
        mem.subnode('Projections', n * ni * np.dtype(self.dtype).itemsize)
        self.pt.estimate_memory(mem.subnode('Projectors'))
