# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""
See Kresse, Phys. Rev. B 54, 11169 (1996)
"""

import numpy as np
from numpy.fft import fftn, ifftn

from gpaw.utilities.blas import axpy
from gpaw.fd_operators import FDOperator
from gpaw.utilities.tools import construct_reciprocal

"""About mixing-related classes.

(FFT/Broyden)BaseMixer: These classes know how to mix one density
array and store history etc.  But they do not take care of complexity
like spin.

(SpinSum/etc.)MixerDriver: These combine one or more BaseMixers to
implement a full algorithm.  Think of them as stateless (immutable).
The user can give an object of these types as input, but they will generally
be constructed by a utility function so the interface is nice.

The density object always wraps the (X)MixerDriver with a
MixerWrapper.  The wrapper contains the common code for all mixers so
we don't have to implement it multiple times (estimate memory, etc.).

In the end, what the user provides is probably a dictionary anyway, and the
relevant objects are instantiated automatically."""


class BaseMixer:
    name = 'pulay'

    """Pulay density mixer."""
    def __init__(self, beta, nmaxold, weight):
        """Construct density-mixer object.

        Parameters:

        beta: float
            Mixing parameter between zero and one (one is most
            aggressive).
        nmaxold: int
            Maximum number of old densities.
        weight: float
            Weight parameter for special metric (for long wave-length
            changes).

        """

        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight

    def initialize_metric(self, gd):
        self.gd = gd

        if self.weight == 1:
            self.metric = None

        else:
            a = 0.125 * (self.weight + 7)
            b = 0.0625 * (self.weight - 1)
            c = 0.03125 * (self.weight - 1)
            d = 0.015625 * (self.weight - 1)
            self.metric = FDOperator([a,
                                      b, b, b, b, b, b,
                                      c, c, c, c, c, c, c, c, c, c, c, c,
                                      d, d, d, d, d, d, d, d],
                                     [(0, 0, 0),  # a
                                      (-1, 0, 0), (1, 0, 0),  # b
                                      (0, -1, 0), (0, 1, 0),
                                      (0, 0, -1), (0, 0, 1),
                                      (1, 1, 0), (1, 0, 1), (0, 1, 1),  # c
                                      (1, -1, 0), (1, 0, -1), (0, 1, -1),
                                      (-1, 1, 0), (-1, 0, 1), (0, -1, 1),
                                      (-1, -1, 0), (-1, 0, -1), (0, -1, -1),
                                      (1, 1, 1), (1, 1, -1), (1, -1, 1),  # d
                                      (-1, 1, 1), (1, -1, -1), (-1, -1, 1),
                                      (-1, 1, -1), (-1, -1, -1)],
                                     gd, float).apply
            self.mR_G = gd.empty()

    def reset(self):
        """Reset Density-history.

        Called at initialization and after each move of the atoms.

        my_nuclei:   All nuclei in local domain.
        """

        # History for Pulay mixing of densities:
        self.nt_iG = []  # Pseudo-electron densities
        self.R_iG = []  # Residuals
        self.A_ii = np.zeros((0, 0))

        self.D_iap = []
        self.dD_iap = []

    def calculate_charge_sloshing(self, R_G):
        return self.gd.integrate(np.fabs(R_G))

    def mix_single_density(self, nt_G, D_ap):
        iold = len(self.nt_iG)

        dNt = np.inf
        if iold > 0:
            if iold > self.nmaxold:
                # Throw away too old stuff:
                del self.nt_iG[0]
                del self.R_iG[0]
                del self.D_iap[0]
                del self.dD_iap[0]
                # for D_p, D_ip, dD_ip in self.D_a:
                #     del D_ip[0]
                #     del dD_ip[0]
                iold = self.nmaxold

            # Calculate new residual (difference between input and
            # output density):
            R_G = nt_G - self.nt_iG[-1]
            dNt = self.calculate_charge_sloshing(R_G)
            self.R_iG.append(R_G)
            self.dD_iap.append([])
            for D_p, D_ip in zip(D_ap, self.D_iap[-1]):
                self.dD_iap[-1].append(D_p - D_ip)

            # Update matrix:
            A_ii = np.zeros((iold, iold))
            i2 = iold - 1

            if self.metric is None:
                mR_G = R_G
            else:
                mR_G = self.mR_G
                self.metric(R_G, mR_G)

            for i1, R_1G in enumerate(self.R_iG):
                a = self.gd.comm.sum(self.dotprod(R_1G, mR_G, self.dD_iap[i1],
                                                  self.dD_iap[-1]))
                A_ii[i1, i2] = a
                A_ii[i2, i1] = a
            A_ii[:i2, :i2] = self.A_ii[-i2:, -i2:]
            self.A_ii = A_ii

            try:
                B_ii = np.linalg.inv(A_ii)
            except np.linalg.LinAlgError:
                alpha_i = np.zeros(iold)
                alpha_i[-1] = 1.0
            else:
                alpha_i = B_ii.sum(1)
                try:
                    # Normalize:
                    alpha_i /= alpha_i.sum()
                except ZeroDivisionError:
                    alpha_i[:] = 0.0
                    alpha_i[-1] = 1.0

            # Calculate new input density:
            nt_G[:] = 0.0
            # for D_p, D_ip, dD_ip in self.D_a:
            for D in D_ap:
                D[:] = 0.0
            beta = self.beta
            for i, alpha in enumerate(alpha_i):
                axpy(alpha, self.nt_iG[i], nt_G)
                axpy(alpha * beta, self.R_iG[i], nt_G)
                for D_p, D_ip, dD_ip in zip(D_ap, self.D_iap[i],
                                            self.dD_iap[i]):
                    axpy(alpha, D_ip, D_p)
                    axpy(alpha * beta, dD_ip, D_p)

        # Store new input density (and new atomic density matrices):
        self.nt_iG.append(nt_G.copy())
        self.D_iap.append([])
        for D_p in D_ap:
            self.D_iap[-1].append(D_p.copy())
        return dNt

    # may presently be overridden by passing argument in constructor
    def dotprod(self, R1_G, R2_G, dD1_ap, dD2_ap):
        return np.vdot(R1_G, R2_G).real

    def estimate_memory(self, mem, gd):
        gridbytes = gd.bytecount()
        mem.subnode('nt_iG, R_iG', 2 * self.nmaxold * gridbytes)

    def __repr__(self):
        classname = self.__class__.__name__
        template = '%s(beta=%f, nmaxold=%d, weight=%f)'
        string = template % (classname, self.beta, self.nmaxold, self.weight)
        return string


class ExperimentalDotProd:
    def __init__(self, calc):
        self.calc = calc

    def __call__(self, R1_G, R2_G, dD1_ap, dD2_ap):
        prod = np.vdot(R1_G, R2_G).real
        setups = self.calc.wfs.setups
        # okay, this is a bit nasty because it depends on dD1_ap
        # and its friend having come from D_asp.values() and the dictionaries
        # not having been modified.  This is probably true... for now.
        avalues = self.calc.density.D_asp.keys()
        for a, dD1_p, dD2_p in zip(avalues, dD1_ap, dD2_ap):
            I4_pp = setups[a].four_phi_integrals()
            dD4_pp = np.outer(dD1_p, dD2_p)  # not sure if corresponds quite
            prod += (I4_pp * dD4_pp).sum()
        return prod


class ReciprocalMetric:
    def __init__(self, weight, k2_Q):
        self.k2_Q = k2_Q
        k2_min = np.min(self.k2_Q)
        self.q1 = (weight - 1) * k2_min

    def __call__(self, R_Q, mR_Q):
        mR_Q[:] = R_Q * (1.0 + self.q1 / self.k2_Q)


class FFTBaseMixer(BaseMixer):
    name = 'fft'

    """Mix the density in Fourier space"""
    def __init__(self, beta, nmaxold, weight):
        BaseMixer.__init__(self, beta, nmaxold, weight)

    def initialize_metric(self, gd):
        self.gd = gd
        k2_Q, N3 = construct_reciprocal(self.gd)

        self.metric = ReciprocalMetric(self.weight, k2_Q)
        self.mR_G = gd.empty(dtype=complex)

    def calculate_charge_sloshing(self, R_Q):
        return self.gd.integrate(np.fabs(ifftn(R_Q).real))

    def mix_single_density(self, nt_G, D_ap):
        # Transform real-space density to Fourier space
        nt_Q = np.ascontiguousarray(fftn(nt_G))

        dNt = BaseMixer.mix_single_density(self, nt_Q, D_ap)

        # Return density in real space
        nt_G[:] = np.ascontiguousarray(ifftn(nt_Q).real)
        return dNt


class BroydenBaseMixer:
    name = 'broyden'

    def __init__(self, beta, nmaxold, weight):
        self.verbose = False
        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = 1.0  # XXX discards argument

    def initialize_metric(self, gd):
        self.gd = gd

    def reset(self):
        self.step = 0
        # self.d_nt_G = []
        # self.d_D_ap = []

        self.R_iG = []
        self.dD_iap = []

        self.nt_iG = []
        self.D_iap = []
        self.c_G = []
        self.v_G = []
        self.u_G = []
        self.u_D = []

    def mix_single_density(self, nt_G, D_ap):
        dNt = np.inf
        if self.step > 2:
            del self.R_iG[0]
            for d_Dp in self.dD_iap:
                del d_Dp[0]
        if self.step > 0:
            self.R_iG.append(nt_G - self.nt_iG[-1])
            for d_Dp, D_p, D_ip in zip(self.dD_iap, D_ap, self.D_iap):
                d_Dp.append(D_p - D_ip[-1])
            fmin_G = self.gd.integrate(self.R_iG[-1] * self.R_iG[-1])
            dNt = self.gd.integrate(np.fabs(self.R_iG[-1]))
            if self.verbose:
                print('Mixer: broyden: fmin_G = %f fmin_D = %f' % fmin_G)
        if self.step == 0:
            self.eta_G = np.empty(nt_G.shape)
            self.eta_D = []
            for D_p in D_ap:
                self.eta_D.append(0)
                self.u_D.append([])
                self.D_iap.append([])
                self.dD_iap.append([])
        else:
            if self.step >= 2:
                del self.c_G[:]
                if len(self.v_G) >= self.nmaxold:
                    del self.u_G[0]
                    del self.v_G[0]
                    for u_D in self.u_D:
                        del u_D[0]
                temp_nt_G = self.R_iG[1] - self.R_iG[0]
                self.v_G.append(temp_nt_G / self.gd.integrate(temp_nt_G *
                                                              temp_nt_G))
                if len(self.v_G) < self.nmaxold:
                    nstep = self.step - 1
                else:
                    nstep = self.nmaxold
                for i in range(nstep):
                    self.c_G.append(self.gd.integrate(self.v_G[i] *
                                                      self.R_iG[1]))
                self.u_G.append(self.beta * temp_nt_G + self.nt_iG[1] -
                                self.nt_iG[0])
                for d_Dp, u_D, D_ip in zip(self.dD_iap, self.u_D, self.D_iap):
                    temp_D_ap = d_Dp[1] - d_Dp[0]
                    u_D.append(self.beta * temp_D_ap + D_ip[1] - D_ip[0])
                usize = len(self.u_G)
                for i in range(usize - 1):
                    a_G = self.gd.integrate(self.v_G[i] * temp_nt_G)
                    axpy(-a_G, self.u_G[i], self.u_G[usize - 1])
                    for u_D in self.u_D:
                        axpy(-a_G, u_D[i], u_D[usize - 1])
            self.eta_G = self.beta * self.R_iG[-1]
            for i, d_Dp in enumerate(self.dD_iap):
                self.eta_D[i] = self.beta * d_Dp[-1]
            usize = len(self.u_G)
            for i in range(usize):
                axpy(-self.c_G[i], self.u_G[i], self.eta_G)
                for eta_D, u_D in zip(self.eta_D, self.u_D):
                    axpy(-self.c_G[i], u_D[i], eta_D)
            axpy(-1.0, self.R_iG[-1], nt_G)
            axpy(1.0, self.eta_G, nt_G)
            for D_p, d_Dp, eta_D in zip(D_ap, self.dD_iap, self.eta_D):
                axpy(-1.0, d_Dp[-1], D_p)
                axpy(1.0, eta_D, D_p)
            if self.step >= 2:
                del self.nt_iG[0]
                for D_ip in self.D_iap:
                    del D_ip[0]
        self.nt_iG.append(np.copy(nt_G))
        for D_ip, D_p in zip(self.D_iap, D_ap):
            D_ip.append(np.copy(D_p))
        self.step += 1
        return dNt


class DummyMixer:
    """Dummy mixer for TDDFT, i.e., it does not mix."""
    name = 'dummy'
    beta = 1.0
    nmaxold = 1
    weight = 1

    def mix(self, basemixers, nt_sG, D_asp):
        return 0.0

    def get_basemixers(self, nspins):
        return []

    def todict(self):
        return {'name': 'dummy'}


class SeparateSpinMixerDriver:
    name = 'separate'

    def __init__(self, basemixerclass, beta, nmaxold, weight):
        self.basemixerclass = basemixerclass

        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight

    def get_basemixers(self, nspins):
        return [self.basemixerclass(self.beta, self.nmaxold, self.weight)
                for _ in range(nspins)]

    def mix(self, basemixers, nt_sG, D_asp):
        """Mix pseudo electron densities."""
        D_asp = D_asp.values()
        D_sap = []
        for s in range(len(nt_sG)):
            D_sap.append([D_sp[s] for D_sp in D_asp])
        dNt = 0.0
        for nt_G, D_ap, basemixer in zip(nt_sG, D_sap, basemixers):
            dNt += basemixer.mix_single_density(nt_G, D_ap)
        return dNt


class SpinSumMixerDriver:
    name = 'sum'
    mix_atomic_density_matrices = False

    def __init__(self, basemixerclass, beta, nmaxold, weight):
        self.basemixerclass = basemixerclass

        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight

    def get_basemixers(self, nspins):
        if nspins == 1:
            raise ValueError('Spin sum mixer expects 2 or 4 components')
        return [self.basemixerclass(self.beta, self.nmaxold, self.weight)]

    def mix(self, basemixers, nt_sG, D_asp):
        assert len(basemixers) == 1
        basemixer = basemixers[0]
        D_asp = D_asp.values()

        collinear = len(nt_sG) == 2

        # Mix density
        if collinear:
            nt_G = nt_sG.sum(0)
        else:
            nt_G = nt_sG[0]

        if self.mix_atomic_density_matrices:
            if collinear:
                D_ap = [D_sp[0] + D_sp[1] for D_sp in D_asp]
            else:
                D_ap = [D_sp[0] for D_sp in D_asp]
            dNt = basemixer.mix_single_density(nt_G, D_ap)
            if collinear:
                dD_ap = [D_sp[0] - D_sp[1] for D_sp in D_asp]
                for D_sp, D_p, dD_p in zip(D_asp, D_ap, dD_ap):
                    D_sp[0] = 0.5 * (D_p + dD_p)
                    D_sp[1] = 0.5 * (D_p - dD_p)
        else:
            dNt = basemixer.mix_single_density(nt_G, D_asp)

        if collinear:
            dnt_G = nt_sG[0] - nt_sG[1]
            # Only new magnetization for spin density
            # dD_ap = [D_sp[0] - D_sp[1] for D_sp in D_asp]

            # Construct new spin up/down densities
            nt_sG[0] = 0.5 * (nt_G + dnt_G)
            nt_sG[1] = 0.5 * (nt_G - dnt_G)

        return dNt


class SpinSumMixerDriver2(SpinSumMixerDriver):
    name = 'sum2'
    mix_atomic_density_matrices = True


class SpinDifferenceMixerDriver:
    name = 'difference'

    def __init__(self, basemixerclass, beta, nmaxold, weight,
                 beta_m=0.7, nmaxold_m=2, weight_m=10.0):
        self.basemixerclass = basemixerclass
        self.beta = beta
        self.nmaxold = nmaxold
        self.weight = weight
        self.beta_m = beta_m
        self.nmaxold_m = nmaxold_m
        self.weight_m = weight_m

    def get_basemixers(self, nspins):
        if nspins == 1:
            raise ValueError('Spin difference mixer expects 2 or 4 components')
        basemixer = self.basemixerclass(self.beta, self.nmaxold, self.weight)
        if nspins == 2:
            basemixer_m = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            return basemixer, basemixer_m
        else:
            basemixer_x = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            basemixer_y = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            basemixer_z = self.basemixerclass(self.beta_m, self.nmaxold_m,
                                              self.weight_m)
            return basemixer, basemixer_x, basemixer_y, basemixer_z

    def mix(self, basemixers, nt_sG, D_asp):
        D_asp = D_asp.values()

        if len(nt_sG) == 2:
            basemixer, basemixer_m = basemixers
        else:
            assert len(nt_sG) == 4
            basemixer, basemixer_x, basemixer_y, basemixer_z = basemixers

        if len(nt_sG) == 2:
            # Mix density
            nt_G = nt_sG.sum(0)
            D_ap = [D_sp[0] + D_sp[1] for D_sp in D_asp]
            dNt = basemixer.mix_single_density(nt_G, D_ap)

            # Mix magnetization
            dnt_G = nt_sG[0] - nt_sG[1]
            dD_ap = [D_sp[0] - D_sp[1] for D_sp in D_asp]
            basemixer_m.mix_single_density(dnt_G, dD_ap)
            # (The latter is not counted in dNt)

            # Construct new spin up/down densities
            nt_sG[0] = 0.5 * (nt_G + dnt_G)
            nt_sG[1] = 0.5 * (nt_G - dnt_G)
            for D_sp, D_p, dD_p in zip(D_asp, D_ap, dD_ap):
                D_sp[0] = 0.5 * (D_p + dD_p)
                D_sp[1] = 0.5 * (D_p - dD_p)
        else:
            # Mix density
            nt_G = nt_sG[0]
            D_ap = [D_sp[0] for D_sp in D_asp]
            dNt = basemixer.mix_single_density(nt_G, D_ap)

            # Mix magnetization
            Dx_ap = [D_sp[1] for D_sp in D_asp]
            Dy_ap = [D_sp[2] for D_sp in D_asp]
            Dz_ap = [D_sp[3] for D_sp in D_asp]
            basemixer_x.mix_single_density(nt_sG[1], Dx_ap)
            basemixer_y.mix_single_density(nt_sG[2], Dy_ap)
            basemixer_z.mix_single_density(nt_sG[3], Dz_ap)

        return dNt


# Dictionaries to get mixers by name:
_backends = {}
_methods = {}
for cls in [FFTBaseMixer, BroydenBaseMixer, BaseMixer]:
    _backends[cls.name] = cls
for cls in [SeparateSpinMixerDriver, SpinSumMixerDriver, SpinSumMixerDriver2,
            SpinDifferenceMixerDriver, DummyMixer]:
    _methods[cls.name] = cls


# This function is used by Density to decide mixer parameters
# that the user did not explicitly provide, i.e., it fills out
# everything that is missing and returns a mixer "driver".
def get_mixer_from_keywords(pbc, nspins, **mixerkwargs):
    # The plan is to first establish a kwargs dictionary with all the
    # defaults, then we update it with values from the user.
    kwargs = {'backend': BaseMixer}

    if np.any(pbc):  # Works on array or boolean
        kwargs.update(beta=0.05, history=5, weight=50.0)
    else:
        kwargs.update(beta=0.25, history=3, weight=1.0)

    if nspins == 1:
        kwargs['method'] = SeparateSpinMixerDriver
    else:
        kwargs['method'] = SpinSumMixerDriver

    # Clean up mixerkwargs (compatibility)
    if 'nmaxold' in mixerkwargs:
        assert 'history' not in mixerkwargs
        mixerkwargs['history'] = mixerkwargs.pop('nmaxold')

    # Now the user override:
    for key in kwargs:
        # Clean any 'None' values out as if they had never been passed:
        val = mixerkwargs.pop(key, None)
        if val is not None:
            kwargs[key] = val

    # Resolve keyword strings (like 'fft') into classes (like FFTBaseMixer):
    driver = _methods.get(kwargs['method'], kwargs['method'])
    baseclass = _backends.get(kwargs['backend'], kwargs['backend'])

    # We forward any remaining mixer kwargs to the actual mixer object.
    # Any user defined variables that do not really exist will cause an error.
    mixer = driver(baseclass, beta=kwargs['beta'],
                   nmaxold=kwargs['history'], weight=kwargs['weight'],
                   **mixerkwargs)
    return mixer


# This is the only object which will be used by Density, sod the others
class MixerWrapper:
    def __init__(self, driver, nspins, gd):
        self.driver = driver

        self.beta = driver.beta
        self.nmaxold = driver.nmaxold
        self.weight = driver.weight
        assert self.weight is not None, driver

        self.basemixers = self.driver.get_basemixers(nspins)
        for basemixer in self.basemixers:
            basemixer.initialize_metric(gd)

    def mix(self, nt_sG, D_asp):
        return self.driver.mix(self.basemixers, nt_sG, D_asp)

    def estimate_memory(self, mem, gd):
        for i, basemixer in enumerate(self.basemixers):
            basemixer.estimate_memory(mem.subnode('Mixer %d' % i), gd)

    def reset(self):
        for basemixer in self.basemixers:
            basemixer.reset()

    def __str__(self):
        lines = ['Density mixing:',
                 'Method: ' + self.driver.name,
                 'Backend: ' + self.driver.basemixerclass.name,
                 'Linear mixing parameter: %g' % self.beta,
                 'Mixing with %d old densities' % self.nmaxold]
        if self.weight == 1:
            lines.append('No damping of long wave oscillations')
        else:
            lines.append('Damping of long wave oscillations: %g' % self.weight)
        return '\n  '.join(lines)


# Helper function to define old-style interfaces to mixers.
# Defines and returns a function which looks like a mixer class
def _definemixerfunc(method, backend):
    def getmixer(beta=None, nmaxold=None, weight=None, **kwargs):
        d = dict(method=method, backend=backend,
                 beta=beta, nmaxold=nmaxold, weight=weight)
        d.update(kwargs)
        return d
    return getmixer


Mixer = _definemixerfunc('separate', 'pulay')
MixerSum = _definemixerfunc('sum', 'pulay')
MixerSum2 = _definemixerfunc('sum2', 'pulay')
MixerDif = _definemixerfunc('difference', 'pulay')
FFTMixer = _definemixerfunc('separate', 'fft')
FFTMixerSum = _definemixerfunc('sum', 'fft')
FFTMixerSum2 = _definemixerfunc('sum2', 'fft')
FFTMixerDif = _definemixerfunc('difference', 'fft')
BroydenMixer = _definemixerfunc('separate', 'broyden')
BroydenMixerSum = _definemixerfunc('sum', 'broyden')
BroydenMixerSum2 = _definemixerfunc('sum2', 'broyden')
BroydenMixerDif = _definemixerfunc('difference', 'broyden')
