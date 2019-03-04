from __future__ import print_function
import functools

from ase.calculators.calculator import Calculator
from ase.data import covalent_radii, atomic_numbers
from ase.units import Bohr, Hartree
from ase.utils import convert_string_to_fd
import numpy as np
from math import pi
from scipy.optimize import minimize
from gpaw.external import ExternalPotential
import copy
'''
Module for constrained DFT
for review see Chem. Rev., 2012, 112 (1), pp 321-370
article on GPAW implementation:
J. Chem. Theory Comput., 2016, 12 (11), pp 5367-5378
'''

class CDFT(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, calc, atoms, charge_regions=[], charges=None,
                 spin_regions=[], spins=None,
                 charge_coefs=None, spin_coefs=None,
                 promolecular_constraint=False, txt='-',
                 minimizer_options={'gtol': 0.01},
                 Rc={}, mu = {'Li': 0.5,'F': 0.7, 'O': 0.7,'V': 0.5},
                 method='BFGS', forces='analytical',
                 use_charge_difference=False,compute_forces=True,
                 maxstep=100, tol=1e-3, bounds=None):

        """Constrained DFT calculator.

        calc: GPAW instance
            DFT calculator object to be constrained.
        charge_regions: list of list of int
            Atom indices of atoms in the different charge_regions.
        spin_regions: list of list of int
            Atom indices of atoms in the different spin_regions.
        charges: list of float
            constrained charges in the different charge_regions.
        spins: list of float
            constrained spins in the different charge_regions.
            Value of 1 sets net magnetisation of one up/alpha electron
        charge_coefs: list of float
            Initial values for charge constraint coefficients (eV).
        spin_coefs: list of float
            Initial values for spin constraint coefficients (eV).
        promolecular_constraint: bool
            Define charge and/or spin constraints from promolecular
            densities, see: dx.doi.org/10.1021/cr200148b Eq. 29-31
            If true, user specified charges/spins are overwritten!
            The atoms (of Atoms object) specifying the charge/spin regions
            need to contain have correct charge/spin state!
            (atoms.set_initial_charges([atomic_charges]) and
            atoms.set_initial_magnetic_moments([moments]))
        txt: None or str or file descriptor
            Log file.  Default id '-' meaning standard out.  Use None for
            no output.
        minimizer_options: dict
            options for scipy optimizers, see:scipy.optimize.minimize
        method: str
            One of scipy optimizers, e.g., BFGS, CG
        forces: str
            cDFT weight function contribution to forces
            'fd' for finite difference or 'analytical'
        difference: bool
            If True, constrain charge difference between two regions
            Then charge_regions needs two regions and charges needs
            only one item which is the charge difference between
            the two regions, the first beign donor, the second acceptor

            If False, each region is treated with the corresponding
            charge constraint
        compute_forces: bool
            Should the forces be computed?
        """

        Calculator.__init__(self)


        self.calc = calc

        self.log = convert_string_to_fd(txt)
        self.method = method
        self.forces = forces
        self.options = minimizer_options
        self.difference = use_charge_difference
        self.compute_forces = compute_forces
        self.Rc = Rc
        self.mu = mu
        # set charge constraints and lagrangians
        self.v_i = np.empty(shape=(0,0))
        self.constraints = np.empty(shape=(0,0))

        self.charge_regions = charge_regions
        self.spin_regions = spin_regions
        self._n_charge_regions = len(self.charge_regions)
        self._n_spin_regions = len(self.spin_regions)
        self._n_regions = self._n_charge_regions + self._n_spin_regions

        self.max_step = maxstep
        self.tol = tol
        self.gtol = minimizer_options['gtol']
        self.bounds = bounds

        if self.bounds is not None:
            self.bounds = np.asarray(self.bounds)/Hartree

        if self.difference:
            # difference calculation only for 2 charge regions
            if self.n_spin_regions != 0 or self._n_charge_regions != 2:
                raise ValueError('No spin constraints '
                    'for charge difference calculations and'
                    ' only two regions allowed')
            assert(self.n_charge_regions == 1)

        if self.n_charge_regions == 0:
            self.regions = []

        else:
            self.charge_i = np.array(charges, dtype=float)

            if charge_coefs is None: # to Hartree
                self.v_i = 0.1 * np.sign(self.charge_i)
            else:
                self.v_i = np.array(charge_coefs) / Hartree

            if not self.difference:
                self.regions = copy.copy(self.charge_regions)

                # The objective is to constrain the number of electrons (nel)
                # in a certain region --> convert charge to nel
                Zn = np.zeros(self.n_charge_regions)
                for j in range(len(Zn)):
                    for atom in atoms[self.charge_regions[j]]:
                        Zn[j] += atom.number

                # combined spin and charge constraints
                self.constraints = Zn - self.charge_i

            else: # constrain charge between two regions
                nD = 0. # neutral donor
                nA = 0. # neutral acceptor

                for atom in atoms[charge_regions[0]]:
                    nD += atom.number
                for atom in atoms[charge_regions[1]]:
                    nA += atom.number

                self.dn_core = nD - nA # difference of core
                self.constraints = [self.dn_core - charges[0]]
                self.regions = charge_regions

        # set spin constraints
        if self.n_spin_regions != 0 and not self.difference:
            spin_i = np.array(spins, dtype=float)
            self.constraints = np.append(self.constraints, spin_i)

            if spin_coefs is None: # to Hartree
                v_is = 0.1 * np.sign(spin_i)
            else:
                v_is = np.array(spin_coefs) / Hartree

            self.v_i = np.append(self.v_i, v_is)
            #self.regions.append(spin_regions)
            [self.regions.append(self.spin_regions[i]) for i in range(self.n_spin_regions)]

        # initialise without v_ext
        atoms.set_calculator(self.calc)
        atoms.get_potential_energy()

        assert atoms.calc.wfs.nspins == 2

        self.cdft_initialised = False

        self.atoms = atoms
        self.gd = self.calc.density.finegd

        if promolecular_constraint:
            self.constraints = get_promolecular_constraints(calc = self.calc,
                atoms=self.atoms, charge_regions=self.charge_regions,
                spin_regions=self.spin_regions,
                charges=charges, spins=spins)

        # get number of core electrons at each constrained region
        # used for pseudo free energy to neglect core contributions
        # in coupling calculation
        self.n_core_electrons = np.zeros((len(self.regions)))
        for a in self.atoms:
            for r in range(len(self.regions[:self.n_charge_regions])):
               if a.index in self.regions[r] and not self.difference:
                  n_core = a.number - self.calc.wfs.setups[a.index].Nv
                  self.n_core_electrons[r] += n_core
               elif a.index in self.regions[r] and self.difference:
                   if r == 0:
                      n_core = a.number - self.calc.wfs.setups[a.index].Nv
                      self.n_core_electrons[r] += n_core
                   else:
                      n_core = a.number - self.calc.wfs.setups[a.index].Nv
                      self.n_core_electrons[r] -= n_core

        w = WeightFunc(self.gd, self.atoms, None, self.Rc, self.mu)
        self.Rc, self.mu = w.get_Rc_and_mu()
        # construct cdft potential
        self.ext = CDFTPotential(regions=self.regions,
                    gd=self.gd,
                    atoms=self.atoms,
                    constraints=self.constraints,
                    n_charge_regions=self.n_charge_regions,
                    difference=self.difference,
                    txt=self.log,
                    vi = self.v_i,
                    Rc=self.Rc, mu=self.mu)

        self.calc.set(external=self.ext)

        self.w = self.ext.w_ig

    def calculate(self, atoms, properties, system_changes):
        # check we're dealing with same atoms
        if atoms != self.atoms:
            self.atoms = atoms

        Calculator.calculate(self, self.atoms)

        # update positions and weight functions
        if 'positions' in system_changes or not self.cdft_initialised:
            self.ext.set_positions_and_atomic_numbers(
                self.atoms.positions / Bohr, self.atoms.numbers)
            self.cdft_initialised = True

        self.atoms.calc = self.calc

        p = functools.partial(print, file=self.log)
        self.iteration = 0
        self.old_v_i = self.v_i.copy()

        def f(v_i):
            #nonlocal iteration
            # very simple step size control
            diff = v_i - self.old_v_i

            if np.any(np.abs(diff) >= self.max_step/Hartree):
                v_i = self.old_v_i + np.sign(v_i - self.old_v_i)*self.max_step/Hartree

            self.ext.set_levels(v_i)
            self.v_i = v_i
            # cDFT free energy <A|H^KS + V_a w_a|A> = Edft + <A|Vw|A>
            self.Ecdft = self.atoms.get_potential_energy() # in eV

            # cDFT corrections to energy
            self.get_atomic_density_correction()
            self.Ecdft += self.get_energy_correction() * Hartree

            # get the cDFT gradient
            dn_i = []
            Delta_n = self.get_energy_correction(return_density=True)

            if self.calc.density.nt_sg is None:
                self.density.interpolate_pseudo_density()

            self.nt_ag = self.calc.density.nt_sg[0]
            self.nt_bg = self.calc.density.nt_sg[1]
            total_electrons = []

            if self.n_charge_regions != 0:
                # total pseudo electron density
                n_gt = self.nt_ag + self.nt_bg

                n_electrons = (self.gd.integrate(self.ext.w_ig[0:self.n_charge_regions]*n_gt,
                   global_integral=True))
                # corrections
                n_electrons += Delta_n[0:self.n_charge_regions]
                # constraint
                diff = n_electrons - self.constraints[0:self.n_charge_regions]
                total_electrons.append(n_electrons)
                dn_i.append(diff)

            if self.n_spin_regions != 0:
                # difference of pseudo spin densities
                Dns_gt = (self.nt_ag - self.nt_bg)
                n_electrons= self.gd.integrate(self.ext.w_ig[self.n_charge_regions:]*Dns_gt,
                   global_integral=True)
                # corrections
                n_electrons += Delta_n[self.n_charge_regions:]
                # constraint
                diff = n_electrons - self.constraints[self.n_charge_regions:]
                total_electrons.append(n_electrons)
                dn_i.append(diff)

            self.dn_i = np.asarray(dn_i)
            self.dn_i = self.dn_i.flatten()
            total_electrons = np.asarray(total_electrons)
            self.w = self.ext.w_ig
            # Do not include external potential
            E_KS = get_ks_energy_wo_external(self.calc)

            self.Edft = E_KS
            # pseudo free energy, neglecting core electrons as done for coupling constant calculation
            if not self.difference:
                self.Ecdft = E_KS + np.dot(self.v_i*Hartree,
                        (np.array(total_electrons)-self.n_core_electrons))

            if self.iteration == 0:
                p('Optimizer: {n}'.format(n=self.method))
                p('Optimizer setups:{n}'.format(n=self.options))

            header = '----------------------------------------\n'
            header += 'Iteration: {0}\n'
            header += 'Energy: {1:10.8f} eV\n'
            header += 'Coefficients [eV]: {2} \n'
            header += 'Errors [e]: {3} \n'

            dn_intermediate = self.dn_i.tolist()
            p(header.format(self.iteration,
                    self.Edft,
                    ''.join('{0:4.3f}  '.format(v) for v in self.v_i * Hartree),
                    ''.join('{0:6.4f}  '.format(dn) for dn in dn_intermediate),
                    fill='left', align='left'))

            self.log.flush()
            self.iteration += 1

            self.old_v_i = self.v_i.copy()
            # Force scipy optimizer to converge when gtol is reached
            if np.all((np.abs(self.dn_i) < self.gtol)):
                return np.zeros(len(self.v_i)),np.zeros(len(self.v_i))

            else:
                return np.abs(self.dn_i), -self.dn_i # return negative because maximising wrt v_i

        def hessian(v_i):
            # Hessian approximated with BFGS
            self.hess = self.update_hessian(v_i)
            return self.hess

        #m = minimize(f, self.v_i, jac=True, bounds=self.bounds, tol = self.tol,method = self.method,
        #             hess = hessian, options=self.options)

        m = minimize(f, self.v_i, jac=True, bounds=self.bounds, tol=self.tol,
                    method=self.method, options=self.options)

        p(str(m.message) + '\n')

        self.density = self.calc.density # TS09-vdw needs this

        self.results['energy'] = self.Edft

        # print to log

        p('Final DFT energy : ' + str(self.Edft) + ' eV \n')
        p('CDFT free energy <A|H+Vw|A> : ' + str(self.Ecdft) + ' eV \n')

        if self.compute_forces:
            f = WeightFunc(self.gd,
                    self.atoms,
                    self.regions, self.Rc, self.mu, new=False)

            f_cdft = f.get_cdft_forces2(dens=self.calc.density,
                v_i=self.v_i,
                n_charge_regions=self.n_charge_regions,
                n_spin_regions=self.n_spin_regions,
                w_ig=self.w,
                method=self.forces,
                difference=self.difference)

            self.calc.wfs.world.broadcast(f_cdft, 0)
            self.ext.set_forces(f_cdft)
            self.results['forces'] = self.atoms.get_forces()

    def get_weight(self, save=True, name='weight', pad=False):
        if not pad:
            w_g = self.w

        else:
            c = CDFTPotential(regions=self.regions, gd=self.gd,
                atoms=self.atoms, constraints=self.constraints,
                n_charge_regions=self.n_charge_regions,
                difference=self.difference, vi=self.v_i,
                Rc=self.Rc, mu=self.mu)

            w_g = c.initialize_partitioning(self.gd, construct=True, pad=True,
                                            global_array=True)
        if save:
            w_s = self.gd.collect(w_g, broadcast=True)

            if self.gd.comm.rank == 0:
                np.save('coarse_weight', w_s)
        return w_g

    def cdft_free_energy(self):
        return self.Ecdft

    def dft_energy(self):
        return self.Edft

    def get_lagrangians(self):
        return self.v_i*Hartree

    def get_constraints(self):
        return self.constraints

    def get_grid(self):
        return self.gd

    def get_all_electron_density(self,gridrefinement=2, spin=None):
        return self.calc.get_all_electron_density(gridrefinement=gridrefinement,
                                                  spin=spin)

    def update_hessian(self,v_i):
        '''Computation of a BFGS Hessian
        returns a pos.def. hessian
        '''
        iteration = self.iteration - 1
        if not self.difference:
            n_regions = len(self.regions)
        else:
            n_regions = 1
        if iteration == 0:
            # Initialize Hessian as identity
            # scaled with gradients
            Hk = np.abs(self.dn_i)*np.identity(n_regions)

        else:
            Hk0 = self.hess
            # Form new Hessian using BFGS
            s = v_i - self.old_v_i
            # difference of gradients = y
            y = self.dn_i - self.old_gradient
            #BFGS step
            #Hk = Hk0 + y*yT/(yT*s) - Hk0*s*sT*Hk0/(sT*Hk0*s)
            #form each term
            first_num = np.dot(y, np.transpose(y))
            first_den = np.dot(np.transpose(y),s)

            second_num = np.dot(Hk0 ,np.dot(s, np.dot(np.transpose(s),Hk0) ) )
            second_den = (np.dot(np.transpose(s), np.dot(Hk0, s)))

            Hk = Hk0 + \
                    first_num/first_den - \
                    second_num/second_den

        #make sure Hk is pos. def.eigs = np.linalg.eigvals(self.Hk)
        hess = Hk.copy()
        eigs = np.linalg.eigvals(hess)
        if any(eigs <= 0.):
            hess = Hk.copy()
            while not all( eig > 0. for eig in eigs):
                #round down smallest eigenvalue with 2 decimals
                mineig = np.floor(min(eigs)*100.)/100.
                hess = hess - mineig*np.identity(n_regions)
                eigs = np.linalg.eigvals(hess)

        self.old_gradient = self.dn_i
        self.old_v_i = v_i
        self.old_hessian = hess

        return hess

    def get_atomic_density_correction(self, return_els=False):
        # eq. 20 of the paper
        self.dn_s = np.zeros((2, len(self.atoms)))

        for i in [0,1]:
            for a, D_sp in self.calc.density.D_asp.items():
                self.dn_s[i,a] += np.sqrt(4.*np.pi)*(np.dot(D_sp[i],
                    self.calc.wfs.setups[a].Delta_pL)[i]\
                    + self.calc.wfs.setups[a].Delta0/2)

        self.gd.comm.sum(self.dn_s)
        for a in range(len(self.atoms)):
            self.dn_s[:,a] += self.atoms[a].number/2.
        if return_els:
            return self.dn_s

    def get_energy_correction(self, return_density=False):
        # Delta n^a part of eq 21

        # for each region
        n_a = np.zeros(len(self.regions))

        # int w_i Dn_i for both spins
        # in spin constraints w_ib = -w_ia
        # inside augmentation spheres w_i = 1

        for c in range(len(self.regions)):
            # sum all atoms in a region
            n_sa = self.dn_s[0,self.regions[c]].sum()
            n_sb = self.dn_s[1,self.regions[c]].sum()
            # total density correction
            n_a[c] = n_sa + n_sb

        for s in range(self.n_spin_regions):
            n_sa = self.dn_s[0, self.regions[self.n_charge_regions+s]].sum()
            n_sb = self.dn_s[1, self.regions[self.n_charge_regions+s]].sum()
            n_a[self.n_charge_regions+s] = n_sa - n_sb

        if return_density:
            if not self.difference:
                # Delta n^a, eq 20
                return n_a
            else:
                # the difference of corrections
                return [n_a[0]-n_a[1]]

        else:
            if not self.difference:
                return (np.dot(self.v_i, n_a))
            else:
                # negative for difference acceptor
                vi_temp = np.array([self.v_i[0],-self.v_i[0]])
                return (np.dot(vi_temp,n_a))


    def get_number_of_electrons_on_atoms(self):
        # return the number of electrons with each gaussian

        nelectrons = []
        ae_dens_correction = self.get_atomic_density_correction(return_els=True)
        if self.calc.density.nt_sg is None:
            self.calc.density.interpolate_pseudo_density()

        dens = self.calc.density.nt_sg[0] + self.calc.density.nt_sg[1]

        for atom in self.atoms:
            # weight function with one atom
            f = WeightFunc(self.gd,
                    self.atoms, [atom.index], self.Rc, self.mu)

            w = f.construct_weight_function()
            n_el = (self.gd.integrate(w*dens,
                   global_integral=True))
            # corrections
            n_el += (ae_dens_correction[:,atom.index]).sum()
            nelectrons.append(n_el)

        return nelectrons

    def write(self,name,mode=None):
        self.calc.write(name,mode=mode)

    def save_parameters(self, name = 'initial',save_weight=True, save_wfs=True):
        if self.gd.comm.rank == 0:
            file = open(name+'.txt','w')
            file.write('NA = %f ,\n' % (self.constraints))
            file.write('FA = %f , \n' % (self.Ecdft))
            file.write('EA = %f , \n' % (self.Edft))
            file.write('Va = %f , \n' % (self.v_i * Hartree))
            file.write('N_charge_regions_A = %d ,\n'  %self.n_charge_regions)
            file.close()

    def get_weight_function_on_coarse_grid(self, regions, gd,
            atoms, constraints, n_charge_regions, difference, save=True):

        gd = self.calc.density.gd
        c = CDFTPotential(regions=self.regions, gd=gd,
                atoms=self.atoms, constraints=self.constraints,
                n_charge_regions=self.n_charge_regions, difference=self.difference,
                vi = self.v_i)

        w_G = c.initialize_partitioning(gd, construct=True)

        if save:
            w_s = gd.collect(w_G, broadcast=True)
            if gd.comm.rank == 0:
                np.save('coarse_weight', w_s)
        return w_G

    def get_coarse_grid(self, save=True):

        if save:
            w_s = self.calc.density.gd.collect(self.gd, broadcast=True)
            if self.calc.density.gd.comm.rank == 0:
                np.save('coarse_grid', w_s)

        return self.calc.density.gd

    @property
    def n_spin_regions(self):
        self._n_spin_regions = len(self.spin_regions)
        return self._n_spin_regions

    @property
    def n_charge_regions(self):
        if not self.difference:
            self._n_charge_regions = len(self.charge_regions)
        else:
            self._n_charge_regions =  1
        return self._n_charge_regions

    @property
    def n_regions(self):
        if not self.difference:
            self._n_regions = self._n_charge_regions + self._n_spin_regions
        else:
            self._n_regions = 2
        return self._n_regions

def gaussians(gd, positions, numbers):
    r_Rv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
    radii = covalent_radii[numbers]
    cutoffs = radii + 3.0
    sigmas = radii * min(covalent_radii) + 0.5
    result_R = gd.zeros()
    for pos, Z, rc, sigma in zip(positions, numbers, cutoffs, sigmas):
        d2_R = ((r_Rv - pos)**2).sum(3)
        a_R = Z / (sigma**3 * (2 * np.pi)**1.5) * np.exp(-d2_R / (2 * sigma**2))
        a_R[d2_R > rc] = 0.0
        result_R += a_R
    return result_R


class CDFTPotential(ExternalPotential):
    def __init__(self, regions, gd,
            atoms, constraints, n_charge_regions, difference, vi, txt='-',
            Rc={}, mu={}):

        self.indices_i = regions
        self.gd = gd
        self.log = convert_string_to_fd(txt)
        self.atoms = atoms
        self.v_i = None
        self.pos_av = None
        self.Z_a = None
        self.w_ig = None
        self.n_charge_regions = n_charge_regions
        self.constraints = constraints
        self.difference = difference
        self.v_i = vi
        self.Rc = Rc
        self.mu = mu
        self.name = 'CDFTPotential'

    def __str__(self):
        self.name = 'CDFTPotential'
        return 'CDFTPotential'

    def get_name(self):
        return self.name

    def update_ae_density(self,ae_dens):
        self.ae_dens = ae_dens

    def get_atoms(self):
        return self.atoms

    def get_ae_density(self):
        return self.ae_dens

    def get_vi(self):
        return self.v_i

    def get_constraints(self):
        return self.constraints

    def set_levels(self, v_i):
        self.v_i = np.array(v_i, dtype=float)
        self.vext_g = None
        return self.v_i

    def set_forces(self, cdft_forces):
        self.cdft_forces = cdft_forces

    def get_cdft_forces(self):
        return self.cdft_forces

    def spin_polarized_potential(self):
        return len(self.constraints) != self.n_charge_regions

    def get_w(self):
        return self.w_ig

    def set_positions_and_atomic_numbers(self, pos_av, Z_a):
        self.pos_av = pos_av
        self.Z_a = Z_a
        self.w_ig = None
        self.vext_g = None

    def initialize_partitioning(self, gd, construct=False, pad=False, global_array=False):
        w = get_all_weight_functions(self.atoms,
                                gd, self.indices_i, self.difference,
                                self.Rc, self.mu)
        if construct:
            return np.array(w)

        self.w_ig = np.array(w)
        p = functools.partial(print, file=self.log)
        p('Number of charge constrained regions: {n}'.format(n=self.n_charge_regions))
        if not self.difference:
           p('Number of spin constrained regions: {n}'.format(n=len(self.indices_i)
            -self.n_charge_regions))
        else:
            p('Number of spin constrained regions: 0')
        p('Charge difference: {n}'.format(n=self.difference))
        p('Parameters')
        p('Atom      Width[A]      Rc[A]')
        for a in self.mu:
            p('  {atom}       {width:.3f}        {Rc:.3f}'.format(atom=a,
                   width=self.mu[a]*Bohr, Rc=self.Rc[a]*Bohr))
        print(file=self.log)
        self.log.flush()

    def calculate_potential(self, gd):
        # return v_ext^{\sigma} = sum_i V_i*w_i^{\sigma}
        if self.w_ig is None:
            self.initialize_partitioning(self.gd)

        pot = self.v_i[:, None] * self.w_ig

        #first alpha spin
        vext_sga = np.sum(np.asarray(pot), axis=0)
        # then beta
        vext_sgb = np.asarray(pot)
        # spin constraints with beta spins
        vext_sgb[self.n_charge_regions:] *= -1.
        vext_sgb = np.sum(vext_sgb, axis=0)
        vext_sg = np.array([vext_sga,vext_sgb])
        # spin-dependent cdft potential
        self.vext_g = vext_sg

    def write(self, writer):
        writer.write(vext='CDFTPotential')

    def read(self,reader):
        pass

    def todict(self):
        return {'name': 'CDFTPotential',
                'regions': self.indices_i,
                'constraints': self.v_i * Hartree,
                'n_charge_regions': self.n_charge_regions,
                'difference': self.difference}

    def get_atomic_hamiltonians(self, setups, atom):
        h_cdft_a = np.zeros(setups.shape)
        h_cdft_b = np.zeros(setups.shape)
        v_i = self.v_i
        if self.difference:
            v_i = [v_i, -v_i]

        for i in range(len(self.indices_i)):
            if atom in self.indices_i[i]:
                h_cdft_a += v_i[i] * 2. * np.sqrt(np.pi)*setups
                h_cdft_b += v_i[i] * 2. * np.sqrt(np.pi)*setups
                if i >= self.n_charge_regions and not self.difference:
                    h_cdft_b *= -1.

        return h_cdft_a, h_cdft_b

    def get_cdft_external_energy(self, dens, nspins, vext_g,
                                vt_g, vbar_g, vt_sg):
        # cDFT works with all-electron (spin)density

        # First the potential
        # spin dependent external potential, array of Vi*wi
        #vext_g = self.calculate_potential(self.gd).copy()
        vt_g += vext_g[0]
        vext_g[1:nspins] = vext_g[1] + vbar_g
        vt_sg[nspins:] = 0.0

        # then energy
        w = self.get_w() #weight functions
        Vi = self.get_vi()
        constraints = self.get_constraints()

        # pseudo electron density on fine grid
        if dens.nt_sg is None:
            dens.interpolate_pseudo_density()

        diff = []

        if self.n_charge_regions != 0:
            # pseudo density
            nt_g = dens.nt_sg[0] + dens.nt_sg[1]
            charge_diff = self.gd.integrate(w[0:self.n_charge_regions],
                           nt_g, global_integral=True) \
                         - constraints[0:self.n_charge_regions]
            diff.append(charge_diff)

        #constrained spins
        if len(constraints) - self.n_charge_regions != 0:
            Delta_nt_g =  dens.nt_sg[0] - dens.nt_sg[1] # pseudo spin difference density
            spin_diff = self.gd.integrate(w[self.n_charge_regions:],
                    Delta_nt_g, global_integral=True) - constraints[self.n_charge_regions:]
            diff.append(spin_diff)

        diff = np.asarray(diff)
        # number of domains
        size = self.gd.comm.size
        return np.dot(Vi, diff/size)[0]

class WeightFunc:
    """ Class which builds a weight function around atoms or molecules
    given the atom index - using normalized Gaussian with cut-off!

    The weight function is constructed on the coarse or fine grid and
    can be used to do charge constraint DFT.

    """
    def __init__(self, gd, atoms, indices, Rc={}, mu={}, new=True):
        """ Given a grid-descriptor, atoms object and an index list
            construct a weight function defined by:
                     n_i(r-R_i)
            w(r) = ---------------
                   sum_a n_a(r-R_a)

            where a runs over all atoms, and i can index
            an atom or a list of atoms comprising a molecule, etc.

            The n_i are construced with atom centered gaussians
            using a pre-defined cut-off Rc_i.

        """
        self.gd = gd
        self.atoms = atoms
        self.indices_i = indices # Indices of constrained charge_regions

        # Weight function parameters in Bohr
        # Cutoffs
        if new:
            new_Rc = {}
            for a in self.atoms:
                if a.symbol in Rc:
                    new_Rc[a.symbol] = Rc[a.symbol] / Bohr
                else:
                    element_number = atomic_numbers[a.symbol]
                    cr = covalent_radii[element_number]
                    #Rc to roughly between 3. and 5.
                    new_Rc[a.symbol] = (cr + 2.5) / Bohr

            self.Rc = new_Rc
        else:
            self.Rc = Rc

        # Construct mu (width) dict
        # mu only sets the width and height so it's in angstrom
        if new:
            new_mu = {}
            for a in self.atoms:
                if a.symbol in mu:
                    new_mu[a.symbol] = mu[a.symbol] / Bohr
                else:
                    element_number = atomic_numbers[a.symbol]
                    cr = covalent_radii[element_number]
                    # mu to be roughly between 0.5 and 1.0 AA
                    cr = (cr * min(covalent_radii) + 0.5)
                    new_mu[a.symbol] = cr / Bohr

            # "Larger" atoms may need a bit more width
            self.mu = new_mu
        else:
            self.mu = mu

    def get_Rc_and_mu(self):
        return self.Rc, self.mu

    def normalized_gaussian(self, dis, mu, Rc):
        # Given mu - width, and Rc
        # a normalized gaussian is constructed
        # around some atom. This is
        # placed on the gd (grid) - and truncated
        # at a given cut-off value Rc. dis
        # are the distances from atom to grid points.

        """ Normalized gaussian is:
                      1
        g(r) = ---------------  e^{-(r-Ra)^2/(2mu^2)}
               mu^3*(2pi)^1.5

        for |r-Ra| <= Rc, 0 elsewhere

        """
        # Check function
        check = abs(dis) <= Rc

        # Make gaussian 3D Guassian
        gauss = 1.0 / (mu * (2.0*pi)**(1./2.)) *\
               np.exp(-dis**2 / (2.0 * mu**2))

        # apply cut-off and return
        return np.array(gauss * check)

    def get_distance_vectors(self, pos, distance = True):
        xyz = self.gd.get_grid_point_distance_vectors(pos)
        if distance:
            # returns vector norm
            return np.sqrt((xyz**2).sum(axis=0))
        else:
            # gives raw vector
            return xyz

    def construct_total_density(self, atoms):
        # Add to empty grid
        dens = self.gd.zeros()

        for atom in atoms:
            charge = atom.number
            symbol = atom.symbol
            pos = atom.position / Bohr

            dis = self.get_distance_vectors(pos)

            dens += charge *\
                     self.normalized_gaussian(dis,
                                              self.mu[symbol],
                                              self.Rc[symbol])
        return dens

    def construct_weight_function(self):
        # Grab atomic / molecular density
        dens_n = self.construct_total_density(self.atoms[self.indices_i])
        # Grab total density
        dens = self.construct_total_density(self.atoms)
        # Check zero elements
        check = dens == 0
        # Add value to zeros ...
        dens += check * 1.0
        # make weight function
        return (dens_n / dens)

    def get_cdft_forces2(self, dens, v_i, n_charge_regions,
            n_spin_regions, w_ig,method, difference):
        ''' Calculate cDFT force as a sum
        dF/dRi = Fi(inside) + Fs(surf)
        due to cutoff (Rc) in gauss
                  / dw(r)
        Fi_a = -V | ------ n(r) dr
                  /  dR_a
        dw(r)
        ----  = sum of Gaussian functions...
        dR_a

        this is computed in get_dw_dRa
        dens = density
        Vc = cDFT constraint value
        method = 'fd' or 'analytical' for
              finite difference or analytical
              dw/dR
        '''
        cdft_forces = np.zeros((len(self.atoms),3))
        prefactor = self.get_derivative_prefactor(n_charge_regions,
                   n_spin_regions,w_ig,v_i,difference)

        if dens.nt_sg is None:
            dens.interpolate_pseudo_density()

        if method == 'analytical':
            dG_dRav = self.get_analytical_gaussian_derivates()
        elif method == 'fd':
            dG_dRav = self.get_fd_gaussian_derivatives()
        else:
            raise NotImplementedError

        for a,atom in enumerate(self.atoms):
            wn_sg = self.gd.zeros()

            # make extended array
            for c in range(n_charge_regions):
                n_g = (dens.nt_sg[0] + dens.nt_sg[1])
                wn_sg += n_g * prefactor[a][0]

            for s in range(n_spin_regions):
                n_g = (dens.nt_sg[0] - dens.nt_sg[1])
                wn_sg += n_g * prefactor[a][1]
            if method == 'LFC':
                # XXX NOT YET WORKING!!!!
                raise NotImplementedError

            else:
                for i in [0,1,2]:
                    cdft_forces[a][i] += -self.gd.integrate(
                        wn_sg * dG_dRav[a][i], global_integral=True)

        return cdft_forces

    def get_fd_gaussian_derivatives(self, dx=1.e-4):
        dG_dRav = {}
        dirs =[[dx,0.,0.],[0.,dx,0.],[0.,0.,dx]]
        for atom in self.atoms:
            dGav = []
            charge = atom.number
            symbol = atom.symbol
            mu = self.mu[symbol]
            Rc = self.Rc[symbol]

            for i in [0,1,2]:
                # move to +dx
                a_posx = atom.position / Bohr + dirs[i]
                a_dis = self.get_distance_vectors(a_posx)
                Ga_posx = charge * self.normalized_gaussian(a_dis, mu, Rc)
                # move to -dx
                a_negx = atom.position / Bohr - dirs[i]
                a_dis = self.get_distance_vectors(a_negx)
                Ga_negx = charge * self.normalized_gaussian(a_dis, mu, Rc)
                # dG/dx
                dGax = (Ga_posx-Ga_negx)/(2*dx)
                dGav.append(dGax)

            dG_dRav[atom.index] = dGav

        return dG_dRav

    def get_derivative_prefactor(self, n_charge_regions, n_spin_regions,
                                w_ig, v_i, difference):

        '''Computes the dw/dRa array needed for derivatives/forces
        eq 31
        needed for lfc-derivative/integrals
        '''

        prefactor = {} # place to store the extended array
        rho_k = self.construct_total_density(self.atoms) # sum_k n_k
        # Check zero elements
        check = rho_k == 0.
        # Add value to zeros for denominator...
        rho_kd = rho_k.copy()
        rho_kd += check * 1.0
        # MAKE AN EXTENDED ARRAY
        for atom in self.atoms:
            wc = self.gd.zeros()
            ws = self.gd.zeros()

            for i in range(n_charge_regions):
                # build V_i [sum_k rho_k + sum_{j in i}rho_i]
                wi = -w_ig[i]

                if not difference:
                    if atom.index in self.indices_i[i]:
                        wi += 1.
                else:
                    if atom.index in self.indices_i[0]:
                        wi += 1.
                    elif atom.index in self.indices_i[1]:
                        wi -= 1.

                wi *= v_i[i]
                wc += wi / rho_kd

            for i in range(n_spin_regions):
                # build V_i [sum_k rho_k + sum_{j in i}rho_i]
                wi = -w_ig[n_charge_regions + i]
                if atom.index in self.indices_i[n_charge_regions + i]:
                    wi += 1.
                wi *= v_i[n_charge_regions + i]
                ws += wi / rho_kd

            prefactor[atom.index] = [wc, ws]
        return prefactor

    def get_analytical_gaussian_derivates(self):
        # equations 32,33,34
        dG_dRav = {} # place to store the extended array

        # MAKE AN EXTENDED ARRAY
        for atom in self.atoms:

            a_pos = atom.position / Bohr
            a_symbol = atom.symbol
            a_charge = atom.number

            a_dis = self.get_distance_vectors(a_pos)
            rRa = -self.get_distance_vectors(a_pos, distance=False)
            dist_rRa = self.get_distance_vectors(a_pos, distance=True)
            check = dist_rRa == 0
            # Add value to zeros ...
            dist_rRa += check * 1.0
            # eq 33
            drRa_dx = rRa[0] / dist_rRa
            drRa_dy = rRa[1] / dist_rRa
            drRa_dz = rRa[2] / dist_rRa

            # Gaussian derivative eq 34

            G_a =  a_charge * \
               self.normalized_gaussian(a_dis,
                   self.mu[a_symbol],
                   self.Rc[a_symbol])

            # within cutoff or at surface ? --> heaviside
            # inside
            check_i = abs(a_dis) <= self.Rc[a_symbol]
            rRc = check_i*a_dis
            dGa_drRa = -rRc * G_a / (self.mu[a_symbol])**2  # (\Theta * (r-R_a) n_A) / \sigma^2

            # at surface

            check_s = abs(abs(a_dis) - self.Rc[a_symbol] ) <= max(self.gd.get_grid_spacings())
            dGa_drRa += check_s * G_a #\ sigma_{A\in i} n_A

            # eq 32

            dGa_dRax = dGa_drRa * drRa_dx
            dGa_dRay = dGa_drRa * drRa_dy
            dGa_dRaz = dGa_drRa * drRa_dz


            dGa_dRav = [dGa_dRax, dGa_dRay, dGa_dRaz]
            dG_dRav[atom.index] = dGa_dRav

        return dG_dRav

def get_ks_energy_wo_external(calc):
    return (calc.hamiltonian.e_kinetic +
        calc.hamiltonian.e_coulomb +
        calc.hamiltonian.e_zero +
        calc.hamiltonian.e_xc -
        calc.hamiltonian.e_entropy)*Hartree

def get_all_weight_functions(atoms, gd, indices_i, difference, Rc, mu, w=[],
                new=False, return_Rc_mu = False):

    for i in range(len(indices_i)):
        wf = WeightFunc(gd, atoms, indices_i[i], Rc, mu, new)
        weig = wf.construct_weight_function()

        if not difference:
            w.append(weig)

        else: # for charge difference constraint
            if i==0:
                w.append(weig)
            else:
                w[0] -= weig # negative for acceptor
    if return_Rc_mu:
        Rc, mu = wf.get_Rc_and_mu()
        return w, Rc, mu
    else:
        return w

def get_promolecular_constraints(calc_a, atoms_a, calc_b, atoms_b,
        charge_constraint=True, spin_constraint=False, restart=False,
        Rc={}, mu={}):

    '''
    - calc_a is for the region you're interested in. Its charge
        should correspond to the "free charge" of the promolecule
    - atoms_a: atoms object for the region of interest
    - calc_b is for the other region
    - atoms_b: atoms object for the other region
    - restart: bool. If true, atoms_a have used calc_a and atoms_b calc_b
    '''

    constraints = []
    atoms = atoms_a + atoms_b

    if not restart:
        atoms_a.set_calculator(calc_a)
        atoms_b.set_calculator(calc_b)
    atoms_a.get_potential_energy()
    atoms_b.get_potential_energy()

    gd = calc_a.density.finegd

    if charge_constraint:
        # can charge and spin constraints be treated at the same time?
        d_a = calc_a.get_all_electron_density(gridrefinement=2, pad=False)
        d_b = calc_b.get_all_electron_density(gridrefinement=2, pad = False)

        charge_region = [atom.index for atom in atoms_a]
        weight = WeightFunc(gd=gd, atoms=atoms, indices=charge_region, Rc=Rc, mu=mu)
        w = weight.construct_weight_function()

        dv = atoms.get_volume() / calc_a.get_number_of_grid_points().prod()
        gridrefinement = 2.
        Nel = (w*(d_a+d_b)).sum() * dv / gridrefinement**3

        Zn = 0
        for atom in atoms_a:
            Zn += atom.number

        constraints.append(Zn - Nel)

    if spin_constraint:
        # can charge and spin constraints be treated at the same time?
        d_a = calc_a.get_all_electron_density(gridrefinement=2, pad=False, spin=0)
        d_b = calc_b.get_all_electron_density(gridrefinement=2, pad = False, spin=1)

        charge_region = [atom.index for atom in atoms_a]
        weight = WeightFunc(gd=gd, atoms=atoms, indices=charge_region, Rc=Rc, mu=mu)
        w = weight.construct_weight_function()

        dv = atoms.get_volume() / calc_a.get_number_of_grid_points().prod()
        gridrefinement = 2.
        Ns = (w*(d_a-d_b)).sum() * dv / gridrefinement**3

        constraints.append(Ns)

    return constraints
