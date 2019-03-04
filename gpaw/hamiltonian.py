# -*- coding: utf-8 -*-
# Copyright (C) 2003-2015  CAMP
# Please see the accompanying LICENSE file for further information.

"""This module defines a Hamiltonian."""
from __future__ import division

import functools

import numpy as np
from ase.units import Ha

from gpaw.arraydict import ArrayDict
from gpaw.external import create_external_potential
from gpaw.hubbard import hubbard
from gpaw.lfc import LFC
from gpaw.poisson import PoissonSolver
from gpaw.spinorbit import soc
from gpaw.transformers import Transformer
from gpaw.utilities import (unpack, pack2, unpack_atomic_matrices,
                            pack_atomic_matrices)
from gpaw.utilities.debug import frozen
from gpaw.utilities.partition import AtomPartition


ENERGY_NAMES = ['e_kinetic', 'e_coulomb', 'e_zero', 'e_external', 'e_xc',
                'e_entropy', 'e_total_free', 'e_total_extrapolated']


def apply_non_local_hamilton(dH_asp, collinear, P, out=None):
    if out is None:
        out = P.new()
    for a, I1, I2 in P.indices:
        if collinear:
            dH_ii = unpack(dH_asp[a][P.spin])
            out.array[:, I1:I2] = np.dot(P.array[:, I1:I2], dH_ii)
        else:
            dH_xp = dH_asp[a]
            # We need the transpose because
            # we are dotting from the left
            dH_ii = unpack(dH_xp[0]).T
            dH_vii = [unpack(dH_p).T for dH_p in dH_xp[1:]]
            out.array[:, 0, I1:I2] = (np.dot(P.array[:, 0, I1:I2],
                                             dH_ii + dH_vii[2]) +
                                      np.dot(P.array[:, 1, I1:I2],
                                             dH_vii[0] - 1j * dH_vii[1]))
            out.array[:, 1, I1:I2] = (np.dot(P.array[:, 1, I1:I2],
                                             dH_ii - dH_vii[2]) +
                                      np.dot(P.array[:, 0, I1:I2],
                                             dH_vii[0] + 1j * dH_vii[1]))
    return out


@frozen
class Hamiltonian:

    def __init__(self, gd, finegd, nspins, collinear, setups, timer, xc, world,
                 redistributor, vext=None):
        self.gd = gd
        self.finegd = finegd
        self.nspins = nspins
        self.collinear = collinear
        self.setups = setups
        self.timer = timer
        self.xc = xc
        self.world = world
        self.redistributor = redistributor

        self.ncomponents = self.nspins if self.collinear else 1 + 3

        self.atomdist = None
        self.dH_asp = None
        self.vt_xG = None
        self.vt_sG = None
        self.vt_vG = None
        self.vHt_g = None
        self.vt_xg = None
        self.vt_sg = None
        self.vt_vg = None
        self.atom_partition = None

        # Energy contributioons that sum up to e_total_free:
        self.e_kinetic = None
        self.e_coulomb = None
        self.e_zero = None
        self.e_external = None
        self.e_xc = None
        self.e_entropy = None

        self.e_total_free = None
        self.e_total_extrapolated = None
        self.e_kinetic0 = None

        self.ref_vt_sG = None
        self.ref_dH_asp = None

        if isinstance(vext, dict):
            vext = create_external_potential(**vext)
        self.vext = vext  # external potential

        self.positions_set = False
        self.spos_ac = None
        self.soc = False

    @property
    def dH(self):
        return functools.partial(apply_non_local_hamilton,
                                 self.dH_asp, self.collinear)

    def update_atomic_hamiltonians(self, value):
        if isinstance(value, dict):
            dtype = complex if self.soc else float
            tmp = self.setups.empty_atomic_matrix(self.ncomponents,
                                                  self.atom_partition, dtype)
            tmp.update(value)
            value = tmp
        assert isinstance(value, ArrayDict) or value is None, type(value)
        if value is not None:
            value.check_consistency()
        self.dH_asp = value

    def __str__(self):
        s = 'Hamiltonian:\n'
        s += ('  XC and Coulomb potentials evaluated on a {0}*{1}*{2} grid\n'
              .format(*self.finegd.N_c))
        s += '  Using the %s Exchange-Correlation functional\n' % self.xc.name
        # We would get the description of the XC functional here,
        # except the thing has probably not been fully initialized yet.
        if self.vext is not None:
            s += '  External potential:\n    {0}\n'.format(self.vext)
        return s

    def summary(self, fermilevel, log):
        log('Energy contributions relative to reference atoms:',
            '(reference = {0:.6f})\n'.format(self.setups.Eref * Ha))

        energies = [('Kinetic:      ', self.e_kinetic),
                    ('Potential:    ', self.e_coulomb),
                    ('External:     ', self.e_external),
                    ('XC:           ', self.e_xc),
                    ('Entropy (-ST):', self.e_entropy),
                    ('Local:        ', self.e_zero)]

        for name, e in energies:
            log('%-14s %+11.6f' % (name, Ha * e))

        log('--------------------------')
        log('Free energy:   %+11.6f' % (Ha * self.e_total_free))
        log('Extrapolated:  %+11.6f' % (Ha * self.e_total_extrapolated))
        log()
        self.xc.summary(log)

        try:
            correction = self.poisson.correction
        except AttributeError:
            pass
        else:
            c = self.poisson.c  # index of axis perpendicular to dipole-layer
            if not self.gd.pbc_c[c]:
                # zero boundary conditions
                vacuum = 0.0
            else:
                v_q = self.pd3.gather(self.vHt_q)
                if self.pd3.comm.rank == 0:
                    axes = (c, (c + 1) % 3, (c + 2) % 3)
                    v_g = self.pd3.ifft(v_q, local=True).transpose(axes)
                    vacuum = v_g[0].mean()
                else:
                    vacuum = np.nan

            wf1 = (vacuum - fermilevel + correction) * Ha
            wf2 = (vacuum - fermilevel - correction) * Ha
            log('Dipole-layer corrected work functions: {:.6f}, {:.6f} eV'
                .format(wf1, wf2))
            log()

    def set_positions_without_ruining_everything(self, spos_ac,
                                                 atom_partition):
        self.spos_ac = spos_ac
        rank_a = atom_partition.rank_a

        # If both old and new atomic ranks are present, start a blank dict if
        # it previously didn't exist but it will needed for the new atoms.
        # XXX what purpose does this serve?  In what case does it happen?
        # How would one even go about figuring it out?  Why does it all have
        # to be so unreadable? -Ask
        #
        if (self.atom_partition is not None and
            self.dH_asp is None and (rank_a == self.gd.comm.rank).any()):
            self.update_atomic_hamiltonians({})

        if self.atom_partition is not None and self.dH_asp is not None:
            self.timer.start('Redistribute')
            self.dH_asp.redistribute(atom_partition)
            self.timer.stop('Redistribute')

        self.atom_partition = atom_partition
        self.atomdist = self.redistributor.get_atom_distributions(spos_ac)

    def set_positions(self, spos_ac, atom_partition):
        self.vbar.set_positions(spos_ac, atom_partition)
        self.xc.set_positions(spos_ac)
        self.set_positions_without_ruining_everything(spos_ac, atom_partition)
        self.positions_set = True

    def initialize(self):
        self.vt_xg = self.finegd.empty(self.ncomponents)
        self.vt_sg = self.vt_xg[:self.nspins]
        self.vt_vg = self.vt_xg[self.nspins:]
        self.vHt_g = self.finegd.zeros()
        self.vt_xG = self.gd.empty(self.ncomponents)
        self.vt_sG = self.vt_xG[:self.nspins]
        self.vt_vG = self.vt_xG[self.nspins:]

    def update(self, density):
        """Calculate effective potential.

        The XC-potential and the Hartree potential are evaluated on
        the fine grid, and the sum is then restricted to the coarse
        grid."""

        self.timer.start('Hamiltonian')
        if self.vt_sg is None:
            with self.timer('Initialize Hamiltonian'):
                self.initialize()

        finegrid_energies = self.update_pseudo_potential(density)
        coarsegrid_e_kinetic = self.calculate_kinetic_energy(density)

        with self.timer('Calculate atomic Hamiltonians'):
            W_aL = self.calculate_atomic_hamiltonians(density)

        atomic_energies = self.update_corrections(density, W_aL)

        # Make energy contributions summable over world:
        finegrid_energies *= self.finegd.comm.size / self.world.size
        coarsegrid_e_kinetic *= self.gd.comm.size / self.world.size
        # (careful with array orderings/contents)
        energies = atomic_energies  # kinetic, coulomb, zero, external, xc
        energies[1:] += finegrid_energies  # coulomb, zero, external, xc
        energies[0] += coarsegrid_e_kinetic  # kinetic

        with self.timer('Communicate'):  # time possible load imbalance
            self.world.sum(energies)

        (self.e_kinetic0, self.e_coulomb, self.e_zero,
         self.e_external, self.e_xc) = energies

        self.timer.stop('Hamiltonian')

    def update_corrections(self, dens, W_aL):
        self.timer.start('Atomic')
        self.update_atomic_hamiltonians(None)  # XXXX

        e_kinetic = 0.0
        e_coulomb = 0.0
        e_zero = 0.0
        e_external = 0.0
        e_xc = 0.0

        D_asp = self.atomdist.to_work(dens.D_asp)
        dtype = complex if self.soc else float
        dH_asp = self.setups.empty_atomic_matrix(self.ncomponents,
                                                 D_asp.partition, dtype)

        for a, D_sp in D_asp.items():
            W_L = W_aL[a]
            setup = self.setups[a]

            if self.nspins == 2:
                D_p = D_sp.sum(0)
            else:
                D_p = D_sp[0]
            dH_p = (setup.K_p + setup.M_p +
                    setup.MB_p + 2.0 * np.dot(setup.M_pp, D_p) +
                    np.dot(setup.Delta_pL, W_L))
            e_kinetic += np.dot(setup.K_p, D_p) + setup.Kc
            e_zero += setup.MB + np.dot(setup.MB_p, D_p)
            e_coulomb += setup.M + np.dot(D_p, (setup.M_p +
                                                np.dot(setup.M_pp, D_p)))

            if self.soc:
                dH_vii = soc(setup, self.xc, D_sp)
                dH_sp = np.zeros_like(D_sp, dtype=complex)
                dH_sp[1:] = pack2(dH_vii)
            else:
                dH_sp = np.zeros_like(D_sp)

            if setup.HubU is not None:
                #assert self.collinear
                eU, dHU_sp = hubbard(setup, D_sp)
                e_xc += eU
                dH_sp += dHU_sp

            dH_sp[:self.nspins] += dH_p

            if self.vext and self.vext.get_name() == 'CDFTPotential':
                # cDFT atomic hamiltonian, eq. 25
                # energy correction added in cDFT main
                h_cdft_a, h_cdft_b = self.vext.get_atomic_hamiltonians(
                    setups=setup.Delta_pL[:,0], atom = a)

                dH_sp[0] += h_cdft_a
                dH_sp[1] += h_cdft_b

            if self.ref_dH_asp:
                assert self.collinear
                dH_sp += self.ref_dH_asp[a]

            dH_asp[a] = dH_sp

        self.timer.start('XC Correction')
        for a, D_sp in D_asp.items():
            e_xc += self.xc.calculate_paw_correction(self.setups[a], D_sp,
                    dH_asp[a], a=a)
        self.timer.stop('XC Correction')

        for a, D_sp in D_asp.items():
            e_kinetic -= (D_sp * dH_asp[a]).sum().real

        self.update_atomic_hamiltonians(self.atomdist.from_work(dH_asp))
        self.timer.stop('Atomic')

        # Make corrections due to non-local xc:
        # self.Enlxc = 0.0  # XXXxcfunc.get_non_local_energy()
        e_kinetic += self.xc.get_kinetic_energy_correction() / self.world.size

        return np.array([e_kinetic, e_coulomb, e_zero, e_external, e_xc])

    def get_energy(self, occ):
        self.e_kinetic = self.e_kinetic0 + occ.e_band
        self.e_entropy = occ.e_entropy

        self.e_total_free = (self.e_kinetic + self.e_coulomb +
                             self.e_external + self.e_zero + self.e_xc +
                             self.e_entropy)
        self.e_total_extrapolated = occ.extrapolate_energy_to_zero_width(
            self.e_total_free)

        if 0:
            print(self.e_total_free,
                  self.e_total_extrapolated,
                  self.e_kinetic,
                  self.e_kinetic0,
                  self.e_coulomb,
                  self.e_external,
                  self.e_zero,
                  self.e_xc,
                  self.e_entropy)
            1 / 0

        return self.e_total_free

    def linearize_to_xc(self, new_xc, density):
        # Store old hamiltonian
        ref_vt_sG = self.vt_sG.copy()
        ref_dH_asp = self.dH_asp.copy()
        self.xc = new_xc
        self.xc.set_positions(self.spos_ac)
        self.update(density)

        ref_vt_sG -= self.vt_sG
        for a, dH_sp in self.dH_asp.items():
            ref_dH_asp[a] -= dH_sp
        self.ref_vt_sG = ref_vt_sG
        self.ref_dH_asp = self.atomdist.to_work(ref_dH_asp)

    def calculate_forces(self, dens, F_av):
        ghat_aLv = dens.ghat.dict(derivative=True)
        nct_av = dens.nct.dict(derivative=True)
        vbar_av = self.vbar.dict(derivative=True)

        self.calculate_forces2(dens, ghat_aLv, nct_av, vbar_av)
        F_coarsegrid_av = np.zeros_like(F_av)

        # Force from compensation charges:
        _Q, Q_aL = dens.calculate_multipole_moments()
        for a, dF_Lv in ghat_aLv.items():
            F_av[a] += np.dot(Q_aL[a], dF_Lv)

        # Force from smooth core charge:
        for a, dF_v in nct_av.items():
            F_coarsegrid_av[a] += dF_v[0]

        # Force from zero potential:
        for a, dF_v in vbar_av.items():
            F_av[a] += dF_v[0]

        self.xc.add_forces(F_av)
        self.gd.comm.sum(F_coarsegrid_av, 0)
        self.finegd.comm.sum(F_av, 0)
        if self.vext:
            if self.vext.get_name()=='CDFTPotential':
                F_av += self.vext.get_cdft_forces()
        F_av += F_coarsegrid_av

    def apply_local_potential(self, psit_nG, Htpsit_nG, s):
        """Apply the Hamiltonian operator to a set of vectors.

        XXX Parameter description is deprecated!

        Parameters:

        a_nG: ndarray
            Set of vectors to which the overlap operator is applied.
        b_nG: ndarray, output
            Resulting H times a_nG vectors.
        kpt: KPoint object
            k-point object defined in kpoint.py.
        calculate_projections: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | a_nG> are calculated.
            When False, existing P_uni are used
        local_part_only: bool
            When True, the non-local atomic parts of the Hamiltonian
            are not applied and calculate_projections is ignored.

        """
        vt_G = self.vt_sG[s]
        if psit_nG.ndim == 3:
            Htpsit_nG += psit_nG * vt_G
        else:
            for psit_G, Htpsit_G in zip(psit_nG, Htpsit_nG):
                Htpsit_G += psit_G * vt_G

    def apply(self, a_xG, b_xG, wfs, kpt, calculate_P_ani=True):
        """Apply the Hamiltonian operator to a set of vectors.

        Parameters:

        a_nG: ndarray
            Set of vectors to which the overlap operator is applied.
        b_nG: ndarray, output
            Resulting S times a_nG vectors.
        wfs: WaveFunctions
            Wave-function object defined in wavefunctions.py
        kpt: KPoint object
            k-point object defined in kpoint.py.
        calculate_P_ani: bool
            When True, the integrals of projector times vectors
            P_ni = <p_i | a_nG> are calculated.
            When False, existing P_ani are used

        """

        wfs.kin.apply(a_xG, b_xG, kpt.phase_cd)
        self.apply_local_potential(a_xG, b_xG, kpt.s)
        shape = a_xG.shape[:-3]
        P_axi = wfs.pt.dict(shape)

        if calculate_P_ani:  # TODO calculate_P_ani=False is experimental
            wfs.pt.integrate(a_xG, P_axi, kpt.q)
        else:
            for a, P_ni in kpt.P_ani.items():
                P_axi[a][:] = P_ni

        for a, P_xi in P_axi.items():
            dH_ii = unpack(self.dH_asp[a][kpt.s])
            P_axi[a] = np.dot(P_xi, dH_ii)
        wfs.pt.add(b_xG, P_axi, kpt.q)

    def get_xc_difference(self, xc, density):
        """Calculate non-selfconsistent XC-energy difference."""
        if density.nt_sg is None:
            density.interpolate_pseudo_density()
        nt_sg = density.nt_sg
        if hasattr(xc, 'hybrid'):
            xc.calculate_exx()
        finegd_e_xc = xc.calculate(density.finegd, nt_sg)
        D_asp = self.atomdist.to_work(density.D_asp)
        atomic_e_xc = 0.0
        for a, D_sp in D_asp.items():
            setup = self.setups[a]
            atomic_e_xc += xc.calculate_paw_correction(setup, D_sp, a=a)
        e_xc = finegd_e_xc + self.world.sum(atomic_e_xc)
        return e_xc - self.e_xc

    def estimate_memory(self, mem):
        nbytes = self.gd.bytecount()
        nfinebytes = self.finegd.bytecount()
        arrays = mem.subnode('Arrays', 0)
        arrays.subnode('vHt_g', nfinebytes)
        arrays.subnode('vt_sG', self.nspins * nbytes)
        arrays.subnode('vt_sg', self.nspins * nfinebytes)
        self.xc.estimate_memory(mem.subnode('XC'))
        self.poisson.estimate_memory(mem.subnode('Poisson'))
        self.vbar.estimate_memory(mem.subnode('vbar'))

    def write(self, writer):
        # Write all eneriges:
        for name in ENERGY_NAMES:
            energy = getattr(self, name)
            if energy is not None:
                energy *= Ha
            writer.write(name, energy)

        writer.write(
            potential=self.gd.collect(self.vt_xG) * Ha,
            atomic_hamiltonian_matrices=pack_atomic_matrices(self.dH_asp) * Ha)

        self.xc.write(writer.child('xc'))

        if hasattr(self.poisson, 'write'):
            self.poisson.write(writer.child('poisson'))

    def read(self, reader):
        h = reader.hamiltonian

        # Read all energies:
        for name in ENERGY_NAMES:
            energy = h.get(name)
            if energy is not None:
                energy /= reader.ha
            setattr(self, name, energy)

        # Read pseudo potential on the coarse grid
        # and broadcast on kpt/band comm:
        self.initialize()
        self.gd.distribute(h.potential / reader.ha, self.vt_xG)

        self.atom_partition = AtomPartition(self.gd.comm,
                                            np.zeros(len(self.setups), int),
                                            name='hamiltonian-init-serial')

        # Read non-local part of hamiltonian
        self.update_atomic_hamiltonians({})
        dH_sP = h.atomic_hamiltonian_matrices / reader.ha

        if self.gd.comm.rank == 0:
            self.update_atomic_hamiltonians(
                unpack_atomic_matrices(dH_sP, self.setups))

        if hasattr(self.poisson, 'read'):
            self.poisson.read(reader)
            self.poisson.set_grid_descriptor(self.finegd)


class RealSpaceHamiltonian(Hamiltonian):
    def __init__(self, gd, finegd, nspins, collinear, setups, timer, xc, world,
                 vext=None,
                 psolver=None, stencil=3, redistributor=None):
        Hamiltonian.__init__(self, gd, finegd, nspins, collinear,
                             setups, timer, xc,
                             world, vext=vext,
                             redistributor=redistributor)

        # Solver for the Poisson equation:
        if psolver is None:
            psolver = {}
        if isinstance(psolver, dict):
            psolver = PoissonSolver(**psolver)
        self.poisson = psolver
        self.poisson.set_grid_descriptor(self.finegd)

        # Restrictor function for the potential:
        self.restrictor = Transformer(self.finegd, self.redistributor.aux_gd,
                                      stencil)
        self.restrict = self.restrictor.apply

        self.vbar = LFC(self.finegd, [[setup.vbar] for setup in setups],
                        forces=True)

        self.vbar_g = None
        self.npoisson = None

    def restrict_and_collect(self, a_xg, b_xg=None, phases=None):
        if self.redistributor.enabled:
            tmp_xg = self.restrictor.apply(a_xg, output=None, phases=phases)
            b_xg = self.redistributor.collect(tmp_xg, b_xg)
        else:
            b_xg = self.restrictor.apply(a_xg, output=b_xg, phases=phases)
        return b_xg

    def __str__(self):
        s = Hamiltonian.__str__(self)

        degree = self.restrictor.nn * 2 - 1
        name = ['linear', 'cubic', 'quintic', 'heptic'][degree // 2]
        s += ('  Interpolation: tri-%s ' % name +
              '(%d. degree polynomial)\n' % degree)
        s += '  Poisson solver: %s' % self.poisson.get_description()
        return s

    def set_positions(self, spos_ac, rank_a):
        Hamiltonian.set_positions(self, spos_ac, rank_a)
        if self.vbar_g is None:
            self.vbar_g = self.finegd.empty()
        self.vbar_g[:] = 0.0
        self.vbar.add(self.vbar_g)

    def update_pseudo_potential(self, dens):
        self.timer.start('vbar')
        e_zero = self.finegd.integrate(self.vbar_g, dens.nt_g,
                                       global_integral=False)

        vt_g = self.vt_sg[0]
        vt_g[:] = self.vbar_g
        self.timer.stop('vbar')

        e_external = 0.0
        if self.vext is not None:
            if self.vext.get_name() == 'CDFTPotential':
                vext_g = self.vext.get_potential(self.finegd).copy()
                e_external += self.vext.get_cdft_external_energy(dens,
                        self.nspins, vext_g, vt_g, self.vbar_g, self.vt_sg)

            else:
                vext_g = self.vext.get_potential(self.finegd)
                vt_g += vext_g
                e_external = self.finegd.integrate(vext_g, dens.rhot_g,
                                               global_integral=False)

        if self.nspins == 2:
            self.vt_sg[1] = vt_g

        self.timer.start('XC 3D grid')
        e_xc = self.xc.calculate(self.finegd, dens.nt_sg, self.vt_sg)
        e_xc /= self.finegd.comm.size
        self.timer.stop('XC 3D grid')

        self.timer.start('Poisson')
        # npoisson is the number of iterations:
        self.npoisson = self.poisson.solve(self.vHt_g, dens.rhot_g,
                                           charge=-dens.charge,
                                           timer=self.timer)
        self.timer.stop('Poisson')

        self.timer.start('Hartree integrate/restrict')
        e_coulomb = 0.5 * self.finegd.integrate(self.vHt_g, dens.rhot_g,
                                                global_integral=False)

        for vt_g in self.vt_sg:
            vt_g += self.vHt_g

        self.timer.stop('Hartree integrate/restrict')
        energies = np.array([e_coulomb, e_zero, e_external, e_xc])
        return energies

    def calculate_kinetic_energy(self, density):
        # XXX new timer item for kinetic energy?
        self.timer.start('Hartree integrate/restrict')
        self.restrict_and_collect(self.vt_sg, self.vt_sG)

        e_kinetic = 0.0
        s = 0
        for vt_G, nt_G in zip(self.vt_sG, density.nt_sG):
            if self.ref_vt_sG is not None:
                vt_G += self.ref_vt_sG[s]

            if s < self.nspins:
                e_kinetic -= self.gd.integrate(vt_G, nt_G - density.nct_G,
                                               global_integral=False)
            else:
                e_kinetic -= self.gd.integrate(vt_G, nt_G,
                                               global_integral=False)
            s += 1
        self.timer.stop('Hartree integrate/restrict')
        return e_kinetic

    def calculate_atomic_hamiltonians(self, dens):
        def getshape(a):
            return sum(2 * l + 1 for l, _ in enumerate(self.setups[a].ghat_l)),
        W_aL = ArrayDict(self.atomdist.aux_partition, getshape, float)
        if self.vext:
            if self.vext.get_name() != 'CDFTPotential':
                vext_g = self.vext.get_potential(self.finegd)
                dens.ghat.integrate(self.vHt_g + vext_g, W_aL)
            else:
                dens.ghat.integrate(self.vHt_g, W_aL)
        else:
            dens.ghat.integrate(self.vHt_g, W_aL)

        return self.atomdist.to_work(self.atomdist.from_aux(W_aL))

    def calculate_forces2(self, dens, ghat_aLv, nct_av, vbar_av):
        if self.nspins == 2:
            vt_G = self.vt_sG.mean(0)
        else:
            vt_G = self.vt_sG[0]

        self.vbar.derivative(dens.nt_g, vbar_av)
        if self.vext:
            if self.vext.get_name() == 'CDFTPotential':
                # CDFT force added in calculate_forces
                dens.ghat.derivative(self.vHt_g, ghat_aLv)
            else:
                vext_g = self.vext.get_potential(self.finegd)
                dens.ghat.derivative(self.vHt_g + vext_g, ghat_aLv)
        else:
            dens.ghat.derivative(self.vHt_g, ghat_aLv)
        dens.nct.derivative(vt_G, nct_av)

    def get_electrostatic_potential(self, dens):
        self.update(dens)

        v_g = self.finegd.collect(self.vHt_g, broadcast=True)
        v_g = self.finegd.zero_pad(v_g)
        if hasattr(self.poisson, 'correction'):
            assert self.poisson.c == 2
            v_g[:, :, 0] = self.poisson.correction
        return v_g
