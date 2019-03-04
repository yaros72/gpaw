from gpaw.hamiltonian import RealSpaceHamiltonian
from gpaw.solvation.poisson import WeightedFDPoissonSolver
from gpaw.fd_operators import Gradient
from gpaw.io.logger import indent
from ase.units import Hartree
import numpy as np


class SolvationRealSpaceHamiltonian(RealSpaceHamiltonian):
    """Realspace Hamiltonian with continuum solvent model.

    See also Section III of
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
    """

    def __init__(
        self,
        # solvation related arguments:
        cavity, dielectric, interactions,
        # RealSpaceHamiltonian arguments:
        gd, finegd, nspins, collinear, setups, timer, xc, world,
        redistributor, vext=None, psolver=None,
        stencil=3):
        """Constructor of SolvationRealSpaceHamiltonian class.

        Additional arguments not present in RealSpaceHamiltonian:
        cavity       -- A Cavity instance.
        dielectric   -- A Dielectric instance.
        interactions -- A list of Interaction instances.
        """
        self.cavity = cavity
        self.dielectric = dielectric
        self.interactions = interactions
        cavity.set_grid_descriptor(finegd)
        dielectric.set_grid_descriptor(finegd)
        for ia in interactions:
            ia.set_grid_descriptor(finegd)
        if psolver is None:
            psolver = WeightedFDPoissonSolver()
        psolver.set_dielectric(self.dielectric)
        self.gradient = None
        RealSpaceHamiltonian.__init__(
            self,
            gd, finegd, nspins, collinear, setups, timer, xc, world,
            vext=vext, psolver=psolver,
            stencil=stencil, redistributor=redistributor)

        for ia in interactions:
            setattr(self, 'e_' + ia.subscript, None)
        self.new_atoms = None
        self.vt_ia_g = None
        self.e_el_free = None
        self.e_el_extrapolated = None

    def __str__(self):
        s = RealSpaceHamiltonian.__str__(self) + '\n'
        s += '  Solvation:\n'
        components = [self.cavity, self.dielectric] + self.interactions
        s += ''.join([indent(str(c), 2) for c in components])
        return s

    def estimate_memory(self, mem):
        RealSpaceHamiltonian.estimate_memory(self, mem)
        solvation = mem.subnode('Solvation')
        for name, obj in [
            ('Cavity', self.cavity),
            ('Dielectric', self.dielectric),
        ] + [('Interaction: ' + ia.subscript, ia) for ia in self.interactions]:
            obj.estimate_memory(solvation.subnode(name))

    def update_atoms(self, atoms, log):
        self.new_atoms = atoms.copy()
        log('Solvation position-dependent initialization:')
        self.cavity.update_atoms(atoms, log)
        for ia in self.interactions:
            ia.update_atoms(atoms, log)

    def initialize(self):
        self.gradient = [
            Gradient(self.finegd, i, 1.0, self.poisson.nn) for i in (0, 1, 2)
        ]
        self.vt_ia_g = self.finegd.zeros()
        self.cavity.allocate()
        self.dielectric.allocate()
        for ia in self.interactions:
            ia.allocate()
        RealSpaceHamiltonian.initialize(self)

    def update(self, density):
        self.timer.start('Hamiltonian')
        if self.vt_sg is None:
            self.timer.start('Initialize Hamiltonian')
            self.initialize()
            self.timer.stop('Initialize Hamiltonian')

        cavity_changed = self.cavity.update(self.new_atoms, density)
        if cavity_changed:
            self.cavity.update_vol_surf()
            self.dielectric.update(self.cavity)

        # e_coulomb, Ebar, Eext, Exc =
        finegd_energies = self.update_pseudo_potential(density)
        self.finegd.comm.sum(finegd_energies)
        ia_changed = [
            ia.update(
                self.new_atoms,
                density,
                self.cavity if cavity_changed else None)
            for ia in self.interactions]
        if np.any(ia_changed):
            self.vt_ia_g.fill(.0)
            for ia in self.interactions:
                if ia.depends_on_el_density:
                    self.vt_ia_g += ia.delta_E_delta_n_g
                if self.cavity.depends_on_el_density:
                    self.vt_ia_g += (ia.delta_E_delta_g_g *
                                     self.cavity.del_g_del_n_g)
        if len(self.interactions) > 0:
            for vt_g in self.vt_sg:
                vt_g += self.vt_ia_g
        Eias = np.array([ia.E for ia in self.interactions])

        Ekin1 = self.gd.comm.sum(self.calculate_kinetic_energy(density))
        W_aL = self.calculate_atomic_hamiltonians(density)
        atomic_energies = self.update_corrections(density, W_aL)
        self.world.sum(atomic_energies)

        energies = atomic_energies
        energies[1:] += finegd_energies
        energies[0] += Ekin1
        (self.e_kinetic0, self.e_coulomb, self.e_zero,
         self.e_external, self.e_xc) = energies

        self.finegd.comm.sum(Eias)

        self.cavity.communicate_vol_surf(self.world)
        for E, ia in zip(Eias, self.interactions):
            setattr(self, 'e_' + ia.subscript, E)

        self.new_atoms = None
        self.timer.stop('Hamiltonian')

    def update_pseudo_potential(self, density):
        ret = RealSpaceHamiltonian.update_pseudo_potential(self, density)
        if not self.cavity.depends_on_el_density:
            return ret
        del_g_del_n_g = self.cavity.del_g_del_n_g
        # XXX optimize numerics
        del_eps_del_g_g = self.dielectric.del_eps_del_g_g
        Veps = -1. / (8. * np.pi) * del_eps_del_g_g * del_g_del_n_g
        Veps *= self.grad_squared(self.vHt_g)
        for vt_g in self.vt_sg:
            vt_g += Veps
        return ret

    def calculate_forces(self, dens, F_av):
        # XXX reorganize
        self.el_force_correction(dens, F_av)
        for ia in self.interactions:
            if self.cavity.depends_on_atomic_positions:
                for a, F_v in enumerate(F_av):
                    del_g_del_r_vg = self.cavity.get_del_r_vg(a, dens)
                    for v in (0, 1, 2):
                        F_v[v] -= self.finegd.integrate(
                            ia.delta_E_delta_g_g * del_g_del_r_vg[v],
                            global_integral=False)
            if ia.depends_on_atomic_positions:
                for a, F_v in enumerate(F_av):
                    del_E_del_r_vg = ia.get_del_r_vg(a, dens)
                    for v in (0, 1, 2):
                        F_v[v] -= self.finegd.integrate(
                            del_E_del_r_vg[v],
                            global_integral=False)
        return RealSpaceHamiltonian.calculate_forces(self, dens, F_av)

    def el_force_correction(self, dens, F_av):
        if not self.cavity.depends_on_atomic_positions:
            return
        del_eps_del_g_g = self.dielectric.del_eps_del_g_g
        fixed = 1 / (8 * np.pi) * del_eps_del_g_g * \
            self.grad_squared(self.vHt_g)  # XXX grad_vHt_g inexact in bmgs
        for a, F_v in enumerate(F_av):
            del_g_del_r_vg = self.cavity.get_del_r_vg(a, dens)
            for v in (0, 1, 2):
                F_v[v] += self.finegd.integrate(
                    fixed * del_g_del_r_vg[v],
                    global_integral=False)

    def get_energy(self, occ):
        self.e_kinetic = self.e_kinetic0 + occ.e_band
        self.e_entropy = occ.e_entropy
        self.e_el_free = (
            self.e_kinetic + self.e_coulomb + self.e_external + self.e_zero +
            self.e_xc + self.e_entropy)
        e_total_free = self.e_el_free
        for ia in self.interactions:
            e_total_free += getattr(self, 'e_' + ia.subscript)
        self.e_total_free = e_total_free
        self.e_total_extrapolated = occ.extrapolate_energy_to_zero_width(
            self.e_total_free)
        self.e_el_extrapolated = occ.extrapolate_energy_to_zero_width(
            self.e_el_free)
        return self.e_total_free

    def grad_squared(self, x):
        # XXX ugly
        gs = np.empty_like(x)
        tmp = np.empty_like(x)
        self.gradient[0].apply(x, gs)
        np.square(gs, gs)
        self.gradient[1].apply(x, tmp)
        np.square(tmp, tmp)
        gs += tmp
        self.gradient[2].apply(x, tmp)
        np.square(tmp, tmp)
        gs += tmp
        return gs

    def summary(self, fermilevel, log):
        self.cavity.summary(log)
        log()
        log('Solvation Energy Contributions:')
        for ia in self.interactions:
            E = Hartree * getattr(self, 'e_' + ia.subscript)
            log('%-14s %+11.6f' % (ia.subscript + ':', E))
        E_el_free = Hartree * self.e_el_free
        E_el_extrapolated = Hartree * self.e_el_extrapolated
        log('%-14s %+11.6f' % ('el (free):', E_el_free))
        log('%-14s %+11.6f' % ('el (extrpol.):', E_el_extrapolated))
        log('el (free) is composed of:')
        RealSpaceHamiltonian.summary(self, fermilevel, log)
