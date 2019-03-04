# -*- coding: utf-8 -*-
# Copyright (C) 2003-2007  CAMP
# Copyright (C) 2007-2008  CAMd
# Please see the accompanying LICENSE file for further information.

"""This module defines a PAW-class.

The central object that glues everything together!"""

import numpy as np
from ase.units import Bohr, Ha

from gpaw.utilities.memory import MemNode, maxrss
from gpaw.external import PointChargePotential
from gpaw.xc import XC


class PAW:
    """ASE-calculator interface.

        The following parameters can be used: nbands, xc, kpts, kpt_refine,
        spinpol, gpts, h, charge, symmetry, width, mixer,
        hund, lmax, fixdensity, convergence, txt, parallel,
        communicator, dtype, softgauss and stencils.

        If you don't specify any parameters, you will get:

        Defaults: neutrally charged, LDA, gamma-point calculation, a
        reasonable grid-spacing, zero Kelvin electronic temperature,
        and the number of bands will be equal to the number of atomic
        orbitals present in the setups. Only occupied bands are used
        in the convergence decision. The calculation will be
        spin-polarized if and only if one or more of the atoms have
        non-zero magnetic moments. Text output will be written to
        standard output.

        For a non-gamma point calculation, the electronic temperature
        will be 0.1 eV (energies are extrapolated to zero Kelvin) and
        all symmetries will be used to reduce the number of
        **k**-points."""

    def linearize_to_xc(self, newxc):
        """Linearize Hamiltonian to difference XC functional.

        Used in real time TDDFT to perform calculations with various kernels.
        """
        if isinstance(newxc, str):
            newxc = XC(newxc)
        self.log('Linearizing xc-hamiltonian to ' + str(newxc))
        newxc.initialize(self.density, self.hamiltonian, self.wfs,
                         self.occupations)
        self.hamiltonian.linearize_to_xc(newxc, self.density)

    def attach(self, function, n=1, *args, **kwargs):
        """Register observer function.

        Call *function* using *args* and
        *kwargs* as arguments.

        If *n* is positive, then
        *function* will be called every *n* iterations + the
        final iteration if it would not be otherwise

        If *n* is negative, then *function* will only be
        called on iteration *abs(n)*.

        If *n* is 0, then *function* will only be called
        on convergence"""

        try:
            slf = function.__self__
        except AttributeError:
            pass
        else:
            if slf is self:
                # function is a bound method of self.  Store the name
                # of the method and avoid circular reference:
                function = function.__func__.__name__

        # Replace self in args with another unique reference
        # to avoid circular reference
        if not hasattr(self, 'self_ref'):
            self.self_ref = object()
        self_ = self.self_ref
        args = tuple([self_ if arg is self else arg for arg in args])

        self.observers.append((function, n, args, kwargs))

    def call_observers(self, iter, final=False):
        """Call all registered callback functions."""
        for function, n, args, kwargs in self.observers:
            call = False
            # Call every n iterations, including the last
            if n > 0:
                if ((iter % n) == 0) != final:
                    call = True
            # Call only on iteration n
            elif n < 0 and not final:
                if iter == abs(n):
                    call = True
            # Call only on convergence
            elif n == 0 and final:
                call = True
            if call:
                if isinstance(function, str):
                    function = getattr(self, function)
                # Replace self reference with self
                self_ = self.self_ref
                args = tuple([self if arg is self_ else arg for arg in args])
                function(*args, **kwargs)

    def get_reference_energy(self):
        return self.wfs.setups.Eref * Ha

    def get_homo_lumo(self, spin=None):
        """Return HOMO and LUMO eigenvalues.

        By default, return the true HOMO-LUMO eigenvalues (spin=None).

        If spin is 0 or 1, return HOMO-LUMO eigenvalues taken among
        only those states with the given spin."""
        return self.wfs.get_homo_lumo(spin) * Ha

    def estimate_memory(self, mem):
        """Estimate memory use of this object."""
        for name, obj in [('Density', self.density),
                          ('Hamiltonian', self.hamiltonian),
                          ('Wavefunctions', self.wfs)]:
            obj.estimate_memory(mem.subnode(name))

    def print_memory_estimate(self, log=None, maxdepth=-1):
        """Print estimated memory usage for PAW object and components.

        maxdepth is the maximum nesting level of displayed components.

        The PAW object must be initialize()'d, but needs not have large
        arrays allocated."""
        # NOTE.  This should work with "--gpaw dry_run=N"
        #
        # However, the initial overhead estimate is wrong if this method
        # is called within a real mpirun/gpaw-python context.
        if log is None:
            log = self.log
        log('Memory estimate:')

        mem_init = maxrss()  # initial overhead includes part of Hamiltonian!
        log('  Process memory now: %.2f MiB' % (mem_init / 1024.0**2))

        mem = MemNode('Calculator', 0)
        mem.indent = '  '
        try:
            self.estimate_memory(mem)
        except AttributeError as m:
            log('Attribute error: %r' % m)
            log('Some object probably lacks estimate_memory() method')
            log('Memory breakdown may be incomplete')
        mem.calculate_size()
        mem.write(log.fd, maxdepth=maxdepth, depth=1)
        log()

    def converge_wave_functions(self):
        """Converge the wave-functions if not present."""

        if self.scf and self.scf.converged:
            if isinstance(self.wfs.kpt_u[0].psit_nG, np.ndarray):
                return
            if self.wfs.kpt_u[0].psit_nG is not None:
                self.wfs.initialize_wave_functions_from_restart_file()
                return

        if not self.initialized:
            self.initialize()

        self.set_positions()

        self.scf.converged = False
        fixed = self.density.fixed
        self.density.fixed = True
        self.calculate(system_changes=[])
        self.density.fixed = fixed

    def diagonalize_full_hamiltonian(self, nbands=None, ecut=None, scalapack=None,
                                     expert=False):
        if not self.initialized:
            self.initialize()
        nbands = self.wfs.diagonalize_full_hamiltonian(
            self.hamiltonian, self.atoms,
            self.occupations, self.log,
            nbands, ecut, scalapack, expert)
        self.parameters.nbands = nbands

    def get_number_of_bands(self):
        """Return the number of bands."""
        return self.wfs.bd.nbands

    def get_xc_functional(self):
        """Return the XC-functional identifier.

        'LDA', 'PBE', ..."""

        return self.parameters.get('xc', 'LDA')
        return self.hamiltonian.xc.name

    def get_number_of_spins(self):
        return self.wfs.nspins

    def get_spin_polarized(self):
        """Is it a spin-polarized calculation?"""
        return self.wfs.nspins == 2

    def get_bz_k_points(self):
        """Return the k-points."""
        return self.wfs.kd.bzk_kc.copy()

    def get_ibz_k_points(self):
        """Return k-points in the irreducible part of the Brillouin zone."""
        return self.wfs.kd.ibzk_kc.copy()

    def get_bz_to_ibz_map(self):
        """Return indices from BZ to IBZ."""
        return self.wfs.kd.bz2ibz_k.copy()

    def get_k_point_weights(self):
        """Weights of the k-points.

        The sum of all weights is one."""

        return self.wfs.kd.weight_k

    def get_pseudo_density(self, spin=None, gridrefinement=1,
                           pad=True, broadcast=True):
        """Return pseudo-density array.

        If *spin* is not given, then the total density is returned.
        Otherwise, the spin up or down density is returned (spin=0 or
        1)."""

        if gridrefinement == 1:
            nt_sG = self.density.nt_sG
            gd = self.density.gd
        elif gridrefinement == 2:
            if self.density.nt_sg is None:
                self.density.interpolate_pseudo_density()
            nt_sG = self.density.nt_sg
            gd = self.density.finegd
        else:
            raise NotImplementedError

        if spin is None:
            if self.density.nspins == 1:
                nt_G = nt_sG[0]
            else:
                nt_G = nt_sG.sum(axis=0)
        else:
            if self.density.nspins == 1:
                nt_G = 0.5 * nt_sG[0]
            else:
                nt_G = nt_sG[spin]

        nt_G = gd.collect(nt_G, broadcast=broadcast)

        if nt_G is None:
            return None

        if pad:
            nt_G = gd.zero_pad(nt_G)

        return nt_G / Bohr**3

    get_pseudo_valence_density = get_pseudo_density  # Don't use this one!

    def get_effective_potential(self, spin=0, pad=True, broadcast=True):
        """Return pseudo effective-potential."""
        vt_G = self.hamiltonian.gd.collect(self.hamiltonian.vt_sG[spin],
                                           broadcast=broadcast)
        if vt_G is None:
            return None

        if pad:
            vt_G = self.hamiltonian.gd.zero_pad(vt_G)
        return vt_G * Ha

    def get_electrostatic_potential(self):
        """Return the electrostatic potential.

        This is the potential from the pseudo electron density and the
        PAW-compensation charges.  So, the electrostatic potential will
        only be correct outside the PAW augmentation spheres.
        """

        ham = self.hamiltonian
        dens = self.density
        self.initialize_positions()
        dens.interpolate_pseudo_density()
        dens.calculate_pseudo_charge()
        return ham.get_electrostatic_potential(dens) * Ha

    def get_pseudo_density_corrections(self):
        """Integrated density corrections.

        Returns the integrated value of the difference between the pseudo-
        and the all-electron densities at each atom.  These are the numbers
        you should add to the result of doing e.g. Bader analysis on the
        pseudo density."""
        if self.wfs.nspins == 1:
            return np.array([self.density.get_correction(a, 0)
                             for a in range(len(self.atoms))])
        else:
            return np.array([[self.density.get_correction(a, spin)
                              for a in range(len(self.atoms))]
                             for spin in range(2)])

    def get_all_electron_density(self, spin=None, gridrefinement=2,
                                 pad=True, broadcast=True, collect=True,
                                 skip_core=False):
        """Return reconstructed all-electron density array."""
        n_sG, gd = self.density.get_all_electron_density(
            self.atoms, gridrefinement=gridrefinement, skip_core=skip_core)
        if spin is None:
            if self.density.nspins == 1:
                n_G = n_sG[0]
            else:
                n_G = n_sG.sum(axis=0)
        else:
            if self.density.nspins == 1:
                n_G = 0.5 * n_sG[0]
            else:
                n_G = n_sG[spin]

        if collect:
            n_G = gd.collect(n_G, broadcast=broadcast)

        if n_G is None:
            return None

        if pad:
            n_G = gd.zero_pad(n_G)

        return n_G / Bohr**3

    def get_fermi_level(self):
        """Return the Fermi-level(s)."""
        eFermi = self.occupations.get_fermi_level()
        if eFermi is not None:
            eFermi *= Ha
        return eFermi

    def get_fermi_levels(self):
        """Return the Fermi-levels in case of fixed-magmom."""
        eFermi_np_array = self.occupations.get_fermi_levels()
        if eFermi_np_array is not None:
            eFermi_np_array *= Ha
        return eFermi_np_array

    def get_fermi_levels_mean(self):
        """Return the mean of th Fermi-levels in case of fixed-magmom."""
        eFermi_mean = self.occupations.get_fermi_levels_mean()
        if eFermi_mean is not None:
            eFermi_mean *= Ha
        return eFermi_mean

    def get_fermi_splitting(self):
        """Return the Fermi-level-splitting in case of fixed-magmom."""
        eFermi_splitting = self.occupations.get_fermi_splitting()
        if eFermi_splitting is not None:
            eFermi_splitting *= Ha
        return eFermi_splitting

    def get_wigner_seitz_densities(self, spin):
        """Get the weight of the spin-density in Wigner-Seitz cells
        around each atom.

        The density assigned to each atom is relative to the neutral atom,
        i.e. the density sums to zero.
        """
        from gpaw.analyse.wignerseitz import wignerseitz
        atom_index = wignerseitz(self.wfs.gd, self.atoms)

        nt_G = self.density.nt_sG[spin]
        weight_a = np.empty(len(self.atoms))
        for a in range(len(self.atoms)):
            # XXX Optimize! No need to integrate in zero-region
            smooth = self.wfs.gd.integrate(np.where(atom_index == a,
                                                    nt_G, 0.0))
            correction = self.density.get_correction(a, spin)
            weight_a[a] = smooth + correction

        return weight_a

    def get_dos(self, spin=0, npts=201, width=None):
        """The total DOS.

        Fold eigenvalues with Gaussians, and put on an energy grid.

        returns an (energies, dos) tuple, where energies are relative to the
        vacuum level for non-periodic systems, and the average potential for
        periodic systems.
        """
        if width is None:
            width = 0.1

        w_k = self.wfs.kd.weight_k
        Nb = self.wfs.bd.nbands
        energies = np.empty(len(w_k) * Nb)
        weights = np.empty(len(w_k) * Nb)
        x = 0
        for k, w in enumerate(w_k):
            energies[x:x + Nb] = self.get_eigenvalues(k, spin)
            weights[x:x + Nb] = w
            x += Nb

        from gpaw.utilities.dos import fold
        return fold(energies, weights, npts, width)

    def get_wigner_seitz_ldos(self, a, spin=0, npts=201, width=None):
        """The Local Density of States, using a Wigner-Seitz basis function.

        Project wave functions onto a Wigner-Seitz box at atom ``a``, and
        use this as weight when summing the eigenvalues."""
        if width is None:
            width = 0.1

        from gpaw.utilities.dos import raw_wignerseitz_LDOS, fold
        energies, weights = raw_wignerseitz_LDOS(self, a, spin)
        return fold(energies * Ha, weights, npts, width)

    def get_orbital_ldos(self, a,
                         spin=0, angular='spdf', npts=201, width=None,
                         nbands=None, spinorbit=False):
        """The Local Density of States, using atomic orbital basis functions.

        Project wave functions onto an atom orbital at atom ``a``, and
        use this as weight when summing the eigenvalues.

        The atomic orbital has angular momentum ``angular``, which can be
        's', 'p', 'd', 'f', or any combination (e.g. 'sdf').

        An integer value for ``angular`` can also be used to specify a specific
        projector function to project onto.

        Setting nbands limits the number of bands included. This speeds up the
        calculation if one has many bands in the calculator but is only
        interested in the DOS at low energies.
        """
        from gpaw.utilities.dos import (raw_orbital_LDOS,
                                        raw_spinorbit_orbital_LDOS, fold)
        if width is None:
            width = 0.1

        if not spinorbit:
            energies, weights = raw_orbital_LDOS(self, a, spin, angular,
                                                 nbands)
        else:
            energies, weights = raw_spinorbit_orbital_LDOS(self, a, spin,
                                                           angular)
        return fold(energies * Ha, weights, npts, width)

    def get_lcao_dos(self, atom_indices=None, basis_indices=None,
                     npts=201, width=None):
        """Get density of states projected onto orbitals in LCAO mode.

        basis_indices is a list of indices of basis functions on which
        to project.  To specify all basis functions on a set of atoms,
        you can supply atom_indices instead.  Both cannot be given
        simultaneously."""

        both_none = atom_indices is None and basis_indices is None
        neither_none = atom_indices is not None and basis_indices is not None
        if both_none or neither_none:
            raise ValueError('Please give either atom_indices or '
                             'basis_indices but not both')

        if width is None:
            width = 0.1

        if self.wfs.S_qMM is None:
            from gpaw.utilities.dos import RestartLCAODOS
            lcaodos = RestartLCAODOS(self)
        else:
            from gpaw.utilities.dos import LCAODOS
            lcaodos = LCAODOS(self)

        if atom_indices is not None:
            basis_indices = lcaodos.get_atom_indices(atom_indices)

        eps_n, w_n = lcaodos.get_subspace_pdos(basis_indices)
        from gpaw.utilities.dos import fold
        return fold(eps_n * Ha, w_n, npts, width)

    def get_all_electron_ldos(self, mol, spin=0, npts=201, width=None,
                              wf_k=None, P_aui=None, lc=None, raw=False):
        """The Projected Density of States, using all-electron wavefunctions.

        Projects onto a pseudo_wavefunctions (wf_k) corresponding to some band
        n and uses P_aui ([paw.nuclei[a].P_uni[:,n,:] for a in atoms]) to
        obtain the all-electron overlaps.
        Instead of projecting onto a wavefunction, a molecular orbital can
        be specified by a linear combination of weights (lc)
        """
        from gpaw.utilities.dos import all_electron_LDOS, fold

        if raw:
            return all_electron_LDOS(self, mol, spin, lc=lc,
                                     wf_k=wf_k, P_aui=P_aui)
        if width is None:
            width = 0.1

        energies, weights = all_electron_LDOS(self, mol, spin,
                                              lc=lc, wf_k=wf_k, P_aui=P_aui)
        return fold(energies * Ha, weights, npts, width)

    def get_pseudo_wave_function(self, band=0, kpt=0, spin=0, broadcast=True,
                                 pad=True, periodic=False):
        """Return pseudo-wave-function array.

        Units: 1/Angstrom^(3/2)
        """
        if self.wfs.mode == 'lcao' and not self.wfs.positions_set:
            self.initialize_positions()

        if pad:
            psit_G = self.get_pseudo_wave_function(band, kpt, spin, broadcast,
                                                   pad=False,
                                                   periodic=periodic)
            if psit_G is None:
                return
            else:
                return self.wfs.gd.zero_pad(psit_G)

        psit_G = self.wfs.get_wave_function_array(band, kpt, spin,
                                                  periodic=periodic)
        if broadcast:
            if not self.wfs.world.rank == 0:
                psit_G = self.wfs.gd.empty(dtype=self.wfs.dtype,
                                           global_array=True)
            self.wfs.world.broadcast(psit_G, 0)
            return psit_G / Bohr**1.5
        elif self.wfs.world.rank == 0:
            return psit_G / Bohr**1.5

    def get_eigenvalues(self, kpt=0, spin=0, broadcast=True):
        """Return eigenvalue array."""
        eps_n = self.wfs.collect_eigenvalues(kpt, spin)
        if broadcast:
            if self.wfs.world.rank != 0:
                eps_n = np.empty(self.wfs.bd.nbands)
            self.wfs.world.broadcast(eps_n, 0)
        return eps_n * Ha

    def get_occupation_numbers(self, kpt=0, spin=0, broadcast=True):
        """Return occupation array."""
        f_n = self.wfs.collect_occupations(kpt, spin)
        if broadcast:
            if self.wfs.world.rank != 0:
                f_n = np.empty(self.wfs.bd.nbands)
            self.wfs.world.broadcast(f_n, 0)
        return f_n

    def get_xc_difference(self, xc):
        if isinstance(xc, (str, dict)):
            xc = XC(xc)
        xc.set_grid_descriptor(self.density.finegd)
        xc.initialize(self.density, self.hamiltonian, self.wfs,
                      self.occupations)
        xc.set_positions(self.spos_ac)
        if xc.orbital_dependent:
            self.converge_wave_functions()
        return self.hamiltonian.get_xc_difference(xc, self.density) * Ha

    def initial_wannier(self, initialwannier, kpointgrid, fixedstates,
                        edf, spin, nbands):
        """Initial guess for the shape of wannier functions.

        Use initial guess for wannier orbitals to determine rotation
        matrices U and C.
        """
        from ase.dft.wannier import rotation_from_projection
        proj_knw = self.get_projections(initialwannier, spin)
        U_kww = []
        C_kul = []
        for fixed, proj_nw in zip(fixedstates, proj_knw):
            U_ww, C_ul = rotation_from_projection(proj_nw[:nbands],
                                                  fixed,
                                                  ortho=True)
            U_kww.append(U_ww)
            C_kul.append(C_ul)

        U_kww = np.asarray(U_kww)
        return C_kul, U_kww

    def get_wannier_localization_matrix(self, nbands, dirG, kpoint,
                                        nextkpoint, G_I, spin):
        """Calculate integrals for maximally localized Wannier functions."""

        # Due to orthorhombic cells, only one component of dirG is non-zero.
        k_kc = self.wfs.kd.bzk_kc
        G_c = k_kc[nextkpoint] - k_kc[kpoint] - G_I

        return self.get_wannier_integrals(spin, kpoint,
                                          nextkpoint, G_c, nbands)

    def get_wannier_integrals(self, s, k, k1, G_c, nbands=None):
        """Calculate integrals for maximally localized Wannier functions."""

        assert s <= self.wfs.nspins
        kpt_rank, u = divmod(k + len(self.wfs.kd.ibzk_kc) * s,
                             len(self.wfs.kpt_u))
        kpt_rank1, u1 = divmod(k1 + len(self.wfs.kd.ibzk_kc) * s,
                               len(self.wfs.kpt_u))
        kpt_u = self.wfs.kpt_u

        # XXX not for the kpoint/spin parallel case
        assert self.wfs.kd.comm.size == 1

        # If calc is a save file, read in tar references to memory
        # For lcao mode just initialize the wavefunctions from the
        # calculated lcao coefficients
        if self.wfs.mode == 'lcao':
            self.wfs.initialize_wave_functions_from_lcao()
        else:
            self.wfs.initialize_wave_functions_from_restart_file()

        # Get pseudo part
        Z_nn = self.wfs.gd.wannier_matrix(kpt_u[u].psit_nG,
                                          kpt_u[u1].psit_nG, G_c, nbands)

        # Add corrections
        self.add_wannier_correction(Z_nn, G_c, u, u1, nbands)

        self.wfs.gd.comm.sum(Z_nn)

        return Z_nn

    def add_wannier_correction(self, Z_nn, G_c, u, u1, nbands=None):
        """
        Calculate the correction to the wannier integrals Z,
        given by (Eq. 27 ref1)::

                          -i G.r
            Z   = <psi | e      |psi >
             nm       n             m

                           __                __
                   ~      \              a  \     a*   a    a
            Z    = Z    +  ) exp[-i G . R ]  )   P   dO    P
             nmx    nmx   /__            x  /__   ni   ii'  mi'

                           a                 ii'

        Note that this correction is an approximation that assumes the
        exponential varies slowly over the extent of the augmentation sphere.

        ref1: Thygesen et al, Phys. Rev. B 72, 125119 (2005)
        """

        if nbands is None:
            nbands = self.wfs.bd.nbands

        P_ani = self.wfs.kpt_u[u].P_ani
        P1_ani = self.wfs.kpt_u[u1].P_ani
        for a, P_ni in P_ani.items():
            P_ni = P_ani[a][:nbands]
            P1_ni = P1_ani[a][:nbands]
            dO_ii = self.wfs.setups[a].dO_ii
            e = np.exp(-2.j * np.pi * np.dot(G_c, self.spos_ac[a]))
            Z_nn += e * np.dot(np.dot(P_ni.conj(), dO_ii), P1_ni.T)

    def get_projections(self, locfun, spin=0):
        """Project wave functions onto localized functions

        Determine the projections of the Kohn-Sham eigenstates
        onto specified localized functions of the format::

          locfun = [[spos_c, l, sigma], [...]]

        spos_c can be an atom index, or a scaled position vector. l is
        the angular momentum, and sigma is the (half-) width of the
        radial gaussian.

        Return format is::

          f_kni = <psi_kn | f_i>

        where psi_kn are the wave functions, and f_i are the specified
        localized functions.

        As a special case, locfun can be the string 'projectors', in which
        case the bound state projectors are used as localized functions.
        """

        wfs = self.wfs

        if locfun == 'projectors':
            f_kin = []
            for kpt in wfs.kpt_u:
                if kpt.s == spin:
                    f_in = []
                    for a, P_ni in kpt.P_ani.items():
                        i = 0
                        setup = wfs.setups[a]
                        for l, n in zip(setup.l_j, setup.n_j):
                            if n >= 0:
                                for j in range(i, i + 2 * l + 1):
                                    f_in.append(P_ni[:, j])
                            i += 2 * l + 1
                    f_kin.append(f_in)
            f_kni = np.array(f_kin).transpose(0, 2, 1)
            return f_kni.conj()

        from gpaw.lfc import LocalizedFunctionsCollection as LFC
        from gpaw.spline import Spline
        from math import factorial as fac

        nkpts = len(wfs.kd.ibzk_kc)
        nbf = np.sum([2 * l + 1 for pos, l, a in locfun])
        f_kni = np.zeros((nkpts, wfs.bd.nbands, nbf), wfs.dtype)

        spos_xc = []
        splines_x = []
        for spos_c, l, sigma in locfun:
            if isinstance(spos_c, int):
                spos_c = self.spos_ac[spos_c]
            spos_xc.append(spos_c)
            alpha = .5 * Bohr**2 / sigma**2
            r = np.linspace(0, 10. * sigma, 500)
            f_g = (fac(l) * (4 * alpha)**(l + 3 / 2.) *
                   np.exp(-alpha * r**2) /
                   (np.sqrt(4 * np.pi) * fac(2 * l + 1)))
            splines_x.append([Spline(l, rmax=r[-1], f_g=f_g)])

        lf = LFC(wfs.gd, splines_x, wfs.kd, dtype=wfs.dtype)
        lf.set_positions(spos_xc)

        assert wfs.gd.comm.size == 1
        k = 0
        f_ani = lf.dict(wfs.bd.nbands)
        for kpt in wfs.kpt_u:
            if kpt.s != spin:
                continue
            lf.integrate(kpt.psit_nG[:], f_ani, kpt.q)
            i1 = 0
            for x, f_ni in f_ani.items():
                i2 = i1 + f_ni.shape[1]
                f_kni[k, :, i1:i2] = f_ni
                i1 = i2
            k += 1

        return f_kni.conj()

    def get_number_of_grid_points(self):
        return self.wfs.gd.N_c

    def get_number_of_iterations(self):
        return self.scf.niter

    def get_number_of_electrons(self):
        return self.wfs.setups.nvalence - self.density.charge

    def get_electrostatic_corrections(self):
        """Calculate PAW correction to average electrostatic potential."""
        dEH_a = np.zeros(len(self.atoms))
        for a, D_sp in self.density.D_asp.items():
            setup = self.wfs.setups[a]
            dEH_a[a] = setup.dEH0 + np.dot(setup.dEH_p, D_sp.sum(0))
        self.wfs.gd.comm.sum(dEH_a)
        return dEH_a * Ha * Bohr**3

    def get_nonselfconsistent_energies(self, type='beefvdw'):
        from gpaw.xc.bee import BEEFEnsemble
        if type not in ['beefvdw', 'mbeef', 'mbeefvdw']:
            raise NotImplementedError('Not implemented for type = %s' % type)
        assert self.scf.converged
        bee = BEEFEnsemble(self)
        x = bee.create_xc_contributions('exch')
        c = bee.create_xc_contributions('corr')
        if type == 'beefvdw':
            return np.append(x, c)
        elif type == 'mbeef':
            return x.flatten()
        elif type == 'mbeefvdw':
            return np.append(x.flatten(), c)

    def embed(self, q_p, rc=0.2, rc2=np.inf, width=1.0):
        """Embed QM region in point-charges."""
        pc = PointChargePotential(q_p, rc=rc, rc2=rc2, width=width)
        self.set(external=pc)
        return pc
