from gpaw.xc.functional import XCFunctional
from gpaw.mpi import world
import numpy as np


class NonLocalFunctional(XCFunctional):
    def __init__(self, xcname):
        self.contributions = []
        self.xcs = {}
        XCFunctional.__init__(self, xcname, 'GLLB')
        self.mix = None
        self.mix_vt_sg = None
        self.old_vt_sg = None
        self.old_H_asp = {}

    def set_mix(self, mix):
        self.mix = mix

    def initialize(self, density, hamiltonian, wfs, occupations):
        self.gd = density.gd  # smooth grid describtor
        self.finegd = density.finegd  # fine grid describtor
        self.nt_sg = density.nt_sg  # smooth density
        self.setups = wfs.setups  # All the setups
        self.nspins = wfs.nspins  # number of spins
        self.wfs = wfs
        self.occupations = occupations
        self.density = density
        self.hamiltonian = hamiltonian
        self.nvalence = wfs.nvalence

        # self.vt_sg = paw.vt_sg # smooth potential
        # self.kpt_u = kpt_u # kpoints object
        # self.interpolate = interpolate # interpolation function
        # self.nuclei = nuclei

        # Is this OK place?
        self.initialize0()

    def set_positions(self, spos_ac):
        for contribution in self.contributions:
            contribution.set_positions(spos_ac)

    def pass_stuff_1d(self, ae):
        self.ae = ae

    def initialize0(self):
        for contribution in self.contributions:
            contribution.initialize()

    def initialize_1d(self):
        for contribution in self.contributions:
            contribution.initialize_1d()

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        if self.nspins == 1:
            self.calculate_spinpaired(e_g, n_sg[0], v_sg[0])
        else:
            self.calculate_spinpolarized(e_g, n_sg, v_sg)
        return gd.integrate(e_g)

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp, a=None,
                                 addcoredensity=True):
        return self.calculate_energy_and_derivatives(setup, D_sp, dEdD_sp, a,
                                                     addcoredensity)

    def calculate_spinpaired(self, e_g, n_g, v_g):
        e_g[:] = 0.0
        if self.mix is None:
            for contribution in self.contributions:
                contribution.calculate_spinpaired(e_g, n_g, v_g)
        else:
            cmix = self.mix
            if self.mix_vt_sg is None:
                self.mix_vt_sg = np.zeros_like(v_g)
                self.old_vt_sg = np.zeros_like(v_g)
                cmix = 1.0
            self.mix_vt_sg[:] = 0.0
            for contribution in self.contributions:
                contribution.calculate_spinpaired(e_g, n_g, self.mix_vt_sg)
            self.mix_vt_sg = (cmix * self.mix_vt_sg +
                              (1.0 - cmix) * self.old_vt_sg)
            v_g += self.mix_vt_sg
            self.old_vt_sg[:] = self.mix_vt_sg

    def calculate_spinpolarized(self, e_g, n_sg, v_sg):
        e_g[:] = 0.0
        for contribution in self.contributions:
            contribution.calculate_spinpolarized(e_g, n_sg, v_sg)

    def calculate_energy_and_derivatives(self, setup, D_sp, H_sp, a,
                                         addcoredensity=True):
        if setup.xc_correction is None:
            return 0.0
        Exc = 0.0
        # We are supposed to add to H_sp, not write directly
        H0_sp = H_sp
        H_sp = H0_sp.copy()
        H_sp[:] = 0.0

        if self.mix is None:
            for contribution in self.contributions:
                Exc += contribution.calculate_energy_and_derivatives(
                    setup, D_sp, H_sp, a, addcoredensity)
        else:
            cmix = self.mix
            if a not in self.old_H_asp:
                self.old_H_asp[a] = H_sp.copy()
                cmix = 1.0

            for contribution in self.contributions:
                Exc += contribution.calculate_energy_and_derivatives(
                    setup, D_sp, H_sp, a, addcoredensity)
            H_sp *= cmix
            H_sp += (1 - cmix) * self.old_H_asp[a]
            self.old_H_asp[a][:] = H_sp.copy()

        # if a == 0:
        #     print('eh', H_sp.sum())
        H0_sp += H_sp
        Exc -= setup.xc_correction.e_xc0
        return Exc

    def get_xc_potential_and_energy_1d(self, v_g):
        Exc = 0.0
        for contribution in self.contributions:
            Exc += contribution.add_xc_potential_and_energy_1d(v_g)
        return Exc

    def get_smooth_xc_potential_and_energy_1d(self, vt_g):
        Exc = 0.0
        for contribution in self.contributions:
            Exc += contribution.add_smooth_xc_potential_and_energy_1d(vt_g)
        return Exc

    def initialize_from_atomic_orbitals(self, basis_functions):
        for contribution in self.contributions:
            contribution.initialize_from_atomic_orbitals(basis_functions)

    def get_extra_setup_data(self, dict):
        for contribution in self.contributions:
            contribution.add_extra_setup_data(dict)

    def add_contribution(self, contribution):
        self.contributions.append(contribution)
        self.xcs[contribution.get_name()] = contribution

    def print_functional(self):
        if world.rank != 0:
            return
        print()
        print("Functional being used consists of")
        print("---------------------------------------------------")
        print("| Weight    | Module           | Description      |")
        print("---------------------------------------------------")
        for contribution in self.contributions:
            print("|%9.3f  | %-17s| %-17s|" %
                  (contribution.weight, contribution.get_name(),
                   contribution.get_desc()))
        print("---------------------------------------------------")
        print()

    def read(self, reader):
        for contribution in self.contributions:
            contribution.read(reader)

    def write(self, writer):
        for contribution in self.contributions:
            contribution.write(writer)

    def heeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeelp(self, olddens):
        for contribution in self.contributions:
            try:
                contribution.heeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeelp(olddens)
            except AttributeError:
                pass
