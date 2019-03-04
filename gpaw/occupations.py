# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Occupation number objects."""

import warnings
from math import pi

import numpy as np
from ase.units import Hartree

from gpaw.utilities import erf
from gpaw.mpi import serial_comm


def create_occupation_number_object(name, **kwargs):
    if name == 'fermi-dirac':
        return FermiDirac(**kwargs)
    if name == 'methfessel-paxton':
        return MethfesselPaxton(**kwargs)
    if name == 'orbital-free':
        return TFOccupations()
    raise ValueError('Unknown occupation number object name: ' + name)


def occupation_numbers(occ, eps_skn, weight_k, nelectrons):
    """Calculate occupation numbers from eigenvalues in eV.

    occ: dict
        Example: {'name': 'fermi-dirac', 'width': 0.05} (width in eV).
    eps_skn: ndarray, shape=(nspins, nibzkpts, nbands)
        Eigenvalues.
    weight_k: ndarray, shape=(nibzkpts,)
        Weights of k-points in IBZ (must sum to 1).
    nelectrons: int or float
        Number of electrons.

    Returns a tuple containing:

    * f_skn (sums to nelectrons)
    * fermi-level
    * magnetic moment
    * entropy as -S*T
    """

    from gpaw.grid_descriptor import GridDescriptor

    occ = create_occupation_number_object(**occ)

    eps_skn = np.asarray(eps_skn) / Hartree
    weight_k = np.asarray(weight_k)
    nspins, nkpts, nbands = eps_skn.shape
    f_skn = np.empty_like(eps_skn)

    class SimpleNamespace:
        """Same as types.SimpleNamespace from Python 3.3+."""
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    wfs = SimpleNamespace(kpt_u=[],
                          nvalence=nelectrons,
                          nspins=nspins,
                          kptband_comm=serial_comm,
                          world=serial_comm,
                          gd=GridDescriptor([4, 4, 4], [1.0, 1.0, 1.0],
                                            comm=serial_comm),
                          bd=SimpleNamespace(nbands=nbands,
                                             collect=lambda x: x,
                                             comm=serial_comm),
                          kd=SimpleNamespace(mynks=nspins * nkpts,
                                             comm=serial_comm,
                                             nspins=nspins,
                                             nibzkpts=nkpts,
                                             weight_k=weight_k,
                                             collect=lambda x, broadcast: x))

    for s in range(nspins):
        for k in range(nkpts):
            kpt = SimpleNamespace(s=s,
                                  weight=weight_k[k] * 2 / nspins,
                                  eps_n=eps_skn[s, k],
                                  f_n=f_skn[s, k])
            wfs.kpt_u.append(kpt)

    occ.calculate(wfs)

    return f_skn, occ.fermilevel, occ.magmom, occ.e_entropy


def findroot(func, x, tol=1e-10):
    """Function used for locating Fermi level."""
    xmin = -np.inf
    xmax = np.inf

    # Try 10 step using the gradient:
    niter = 0
    while True:
        f, dfdx = func(x)
        if abs(f) < tol:
            return x, niter
        if f < 0.0 and x > xmin:
            xmin = x
        elif f > 0.0 and x < xmax:
            xmax = x
        dx = -f / max(dfdx, 1e-18)
        if niter == 10 or abs(dx) > 0.01 or not (xmin < x + dx < xmax):
            break  # try bisection
        x += dx
        niter += 1

    # Bracket the solution:
    if not np.isfinite(xmin):
        xmin = x
        fmin = f
        step = 0.01
        while fmin > tol:
            xmin -= step
            fmin = func(xmin)[0]
            step *= 2

    if not np.isfinite(xmax):
        xmax = x
        fmax = f
        step = 0.01
        while fmax < 0:
            xmax += step
            fmax = func(xmax)[0]
            step *= 2

    # Bisect:
    while True:
        x = (xmin + xmax) / 2
        f = func(x)[0]
        if abs(f) < tol:
            return x, niter
        if f > 0:
            xmax = x
        else:
            xmin = x
        niter += 1


class OccupationNumbers:
    """Base class for all occupation number objects."""
    def __init__(self, fixmagmom):
        self.fixmagmom = fixmagmom
        self.magmom = None  # magnetic moment
        self.e_entropy = None  # -ST
        self.e_band = None  # band energy (sum_n eps_n * f_n)
        self.fermilevel = np.nan  # Fermi level
        self.nvalence = None  # number of electrons
        self.split = 0.0  # splitting of Fermi levels from fixmagmom=True
        self.niter = 0  # number of iterations for finding Fermi level
        self.ready = False
        self.fixed_fermilevel = False

    def write(self, writer):
        writer.write(fermilevel=self.fermilevel * Hartree,
                     split=self.split * Hartree)

    def read(self, reader):
        o = reader.occupations
        self.fermilevel = o.fermilevel / reader.ha
        self.split = o.split / reader.ha

    def extrapolate_energy_to_zero_width(self, e_free):
        return e_free

    def calculate(self, wfs):
        """Calculate everything.

        The following is calculated:

        * occupation numbers
        * magnetic moment
        * entropy
        * band energy
        * Fermi level
        """

        # Allow subclasses to adjust nvalence:
        self.set_number_of_electrons(wfs)

        # Allocate:
        for kpt in wfs.kpt_u:
            if kpt.f_n is None:
                kpt.f_n = wfs.bd.empty()

            # There are no eigenvalues, might as well return
            if kpt.eps_n is None:
                return

            # Sanity check.  This class will typically be the first to
            # suffer if any NaNs sneak in.
            assert not np.isnan(kpt.eps_n).any()

        fermilevel = self.fermilevel  # save for later

        # Let the master domain do the work and broadcast results:
        data = np.empty(5)
        if wfs.gd.comm.rank == 0:
            self.calculate_occupation_numbers(wfs)
            self.calculate_band_energy(wfs)
            data[:] = [self.magmom, self.e_entropy, self.e_band,
                       self.fermilevel, self.split]
        wfs.world.broadcast(data, 0)
        (self.magmom, self.e_entropy, self.e_band,
         self.fermilevel, self.split) = data

        for kpt in wfs.kpt_u:
            wfs.gd.comm.broadcast(kpt.f_n, 0)

        if self.fixed_fermilevel:
            self.fermilevel = fermilevel

    def set_number_of_electrons(self, wfs):
        self.nvalence = wfs.nvalence
        self.ready = True

    def calculate_occupation_numbers(self, wfs):
        raise NotImplementedError

    def calculate_band_energy(self, wfs):
        """Sum up all eigenvalues weighted with occupation numbers"""
        e_band = 0.0
        for kpt in wfs.kpt_u:
            e_band += np.dot(kpt.f_n, kpt.eps_n)
        self.e_band = wfs.kptband_comm.sum(e_band)

    def get_fermi_level(self):
        raise ValueError('Can not calculate Fermi level!')


class ZeroKelvin(OccupationNumbers):
    def __init__(self, fixmagmom):
        self.width = 0.0
        OccupationNumbers.__init__(self, fixmagmom)

    def calculate_occupation_numbers(self, wfs):
        if wfs.nspins == 1:
            self.spin_paired(wfs)
        elif self.fixmagmom:
            assert wfs.kd.gamma
            self.fixed_moment(wfs)
        else:
            assert wfs.kd.nibzkpts == 1
            self.spin_polarized(wfs)

        self.e_entropy = 0.0

    def occupy(self, f_n, eps_n, ne, weight=1):
        """Fill in occupation numbers.

        return HOMO and LUMO energies."""

        N = len(f_n)
        if ne == N * weight:
            f_n[:] = weight
            return eps_n[-1], np.inf

        n, f = divmod(ne, weight)
        n = int(n)
        f_n[:n] = weight
        assert n < N
        f_n[n] = f
        f_n[n + 1:] = 0.0
        if f > 0.0:
            return eps_n[n], eps_n[n]
        return eps_n[n - 1], eps_n[n]

    def summary(self, log):
        if np.isfinite(self.fermilevel):
            if self.fixed_fermilevel:
                log('Fixed ', end='')
            if not self.fixmagmom:
                log('Fermi level: %.5f\n' % (Hartree * self.fermilevel))
            else:
                log('Fermi levels: %.5f, %.5f\n' %
                    (Hartree * (self.fermilevel + 0.5 * self.split),
                     Hartree * (self.fermilevel - 0.5 * self.split)))

    def get_fermi_level(self):
        """This function returns the calculated fermi-level.

        Care: you get two distinct fermi-levels if you do
        fixed-magmom calculations. Therefore you should use
        "get_fermi_levels" or "get_fermi_levels_mean" in
        conjunction with "get_fermi_splitting" if you do
        fixed-magmom calculations. We will issue an warning
        otherwise.

        """
        if not np.isfinite(self.fermilevel):
            OccupationNumbers.get_fermi_level(self)  # fail
        else:
            if self.fixmagmom:
                warnings.warn('Please use get_fermi_levels when ' +
                              'using fixmagmom', DeprecationWarning)
                fermilevels = np.empty(2)
                fermilevels[0] = self.fermilevel + 0.5 * self.split
                fermilevels[1] = self.fermilevel - 0.5 * self.split
                return fermilevels
            else:
                return self.fermilevel

    def get_fermi_levels(self):
        """Getting fermi-levels in case of fixed-magmom."""
        if not np.isfinite(self.fermilevel):
            OccupationNumbers.get_fermi_level(self)  # fail
        else:
            if self.fixmagmom:
                fermilevels = np.empty(2)
                fermilevels[0] = self.fermilevel + 0.5 * self.split
                fermilevels[1] = self.fermilevel - 0.5 * self.split
                return fermilevels
            else:
                raise ValueError('Distinct fermi-levels are only vaild ' +
                                 'for fixed-magmom calculations!')

    def get_fermi_levels_mean(self):
        if not np.isfinite(self.fermilevel):
            OccupationNumbers.get_fermi_level(self)  # fail
        else:
            return self.fermilevel

    def get_fermi_splitting(self):
        """Return the splitting of the fermi level in hartree.

        Returns 0.0 if calculation is not done using
        fixmagmom.

        """
        if self.fixmagmom:
            return self.split
        else:
            return 0.0

    def fixed_moment(self, wfs):
        assert wfs.nspins == 2 and wfs.kd.nbzkpts == 1
        fermilevels = np.zeros(2)
        for kpt in wfs.kpt_u:
            eps_n = wfs.bd.collect(kpt.eps_n)
            if eps_n is None:
                f_n = None
            else:
                f_n = wfs.bd.empty(global_array=True)
                sign = 1 - kpt.s * 2
                ne = 0.5 * (self.nvalence + sign * self.magmom)

                homo, lumo = self.occupy(f_n, eps_n, ne)

                fermilevels[kpt.s] = 0.5 * (homo + lumo)
            wfs.bd.distribute(f_n, kpt.f_n)
        wfs.kptband_comm.sum(fermilevels)
        self.fermilevel = fermilevels.mean()
        self.split = fermilevels[0] - fermilevels[1]

    def spin_paired(self, wfs):
        homo = -np.inf
        lumo = np.inf
        if wfs.collinear:
            ne = 0.5 * self.nvalence
        else:
            ne = self.nvalence
        for kpt in wfs.kpt_u:
            eps_n = wfs.bd.collect(kpt.eps_n)
            if wfs.bd.comm.rank == 0:
                f_n = wfs.bd.empty(global_array=True)
                hom, lum = self.occupy(f_n, eps_n,
                                       ne * kpt.weight, kpt.weight)
                homo = max(homo, hom)
                lumo = min(lumo, lum)
            else:
                f_n = None
                self.fermilevel = np.nan
            wfs.bd.distribute(f_n, kpt.f_n)

        if wfs.bd.comm.rank == 0:
            homo = wfs.kd.comm.max(homo)
            lumo = wfs.kd.comm.min(lumo)
            self.fermilevel = 0.5 * (homo + lumo)

        self.magmom = 0.0

    def spin_polarized(self, wfs):
        eps_un = [wfs.bd.collect(kpt.eps_n) for kpt in wfs.kpt_u]
        self.fermilevel = np.nan
        nbands = wfs.bd.nbands
        if wfs.bd.comm.rank == 0:
            if wfs.kd.comm.size == 2:
                if wfs.kd.comm.rank == 1:
                    wfs.kd.comm.send(eps_un[0], 0)
                else:
                    eps_sn = [eps_un[0], np.empty(nbands)]
                    wfs.kd.comm.receive(eps_sn[1], 1)
            else:
                eps_sn = eps_un

            if wfs.kd.comm.rank == 0:
                eps_n = np.ravel(eps_sn)
                f_n = np.empty(nbands * 2)
                nsorted = eps_n.argsort()
                homo, lumo = self.occupy(f_n, eps_n[nsorted], self.nvalence)
                f_sn = f_n[nsorted.argsort()].reshape((2, nbands))
                self.magmom = f_sn[0].sum() - f_sn[1].sum()
                self.fermilevel = 0.5 * (homo + lumo)

            if wfs.kd.comm.size == 2:
                if wfs.kd.comm.rank == 0:
                    wfs.kd.comm.send(f_sn[1], 1)
                else:
                    f_sn = [None, np.empty(nbands)]
                    wfs.kd.comm.receive(f_sn[1], 0)
        else:
            f_sn = [None, None]

        for kpt in wfs.kpt_u:
            wfs.bd.distribute(f_sn[kpt.s], kpt.f_n)


class SmoothDistribution(ZeroKelvin):
    """Base class for Fermi-Dirac and other smooth distributions."""
    def __init__(self, width, fixmagmom):
        """Smooth distribution.

        Find the Fermi level by integrating in energy until
        the number of electrons is correct.

        width: float
            Width of distribution in eV.
        fixmagmom: bool
            Fix spin moment calculations.  A separate Fermi level for
            spin up and down electrons is found: self.fermilevel +
            self.split and self.fermilevel - self.split.
        """

        ZeroKelvin.__init__(self, fixmagmom)
        self.width = width / Hartree

    def todict(self):
        dct = {'width': self.width * Hartree}
        if self.fixmagmom:
            dct['fixmagmom'] = True
        return dct

    def __str__(self):
        s = 'Occupation numbers:\n'
        if self.fixmagmom:
            s += '  Fixed magnetic moment\n'
        if self.fixed_fermilevel:
            s += '  Fixed Fermi level\n'
        return s

    def calculate_occupation_numbers(self, wfs):
        if self.width == 0 or self.nvalence == wfs.bd.nbands * 2:
            ZeroKelvin.calculate_occupation_numbers(self, wfs)
            return

        if not np.isfinite(self.fermilevel):
            self.fermilevel = self.guess_fermi_level(wfs)

        if not self.fixmagmom or wfs.nspins == 1:
            result = self.find_fermi_level(wfs, self.nvalence, self.fermilevel)
            self.fermilevel, self.magmom, self.e_entropy = result

            if wfs.nspins == 1:
                self.magmom = 0.0
        else:
            fermilevels = np.empty(2)
            self.e_entropy = 0.0
            for s in range(2):
                sign = 1 - s * 2
                ne = 0.5 * (self.nvalence + sign * self.magmom)
                fermilevel = self.fermilevel + 0.5 * sign * self.split
                fermilevels[s], magmom, e_entropy = \
                    self.find_fermi_level(wfs, ne, fermilevel, [s])
                self.e_entropy += e_entropy
            self.fermilevel = fermilevels.mean()
            self.split = fermilevels[0] - fermilevels[1]

    def guess_fermi_level(self, wfs):
        fermilevel = 0.0

        kd = wfs.kd

        myeps_un = np.empty((kd.mynks, wfs.bd.nbands))
        for u, kpt in enumerate(wfs.kpt_u):
            myeps_un[u] = wfs.bd.collect(kpt.eps_n)

        if wfs.bd.comm.rank == 0:
            eps_skn = kd.collect(myeps_un, broadcast=False)
            if kd.comm.rank == 0:
                eps_n = eps_skn.ravel()
                w_skn = np.empty((kd.nspins, kd.nibzkpts, wfs.bd.nbands))
                w_skn[:] = (2.0 / wfs.nspins * kd.weight_k[:, np.newaxis])
                w_n = w_skn.ravel()
                n_i = eps_n.argsort()
                w_i = w_n[n_i]
                f_i = np.add.accumulate(w_i) - 0.5 * w_i
                i = np.nonzero(f_i >= self.nvalence)[0][0]
                if i == 0:
                    fermilevel = eps_n[n_i[0]]
                else:
                    fermilevel = ((eps_n[n_i[i]] *
                                   (self.nvalence - f_i[i - 1]) +
                                   eps_n[n_i[i - 1]] *
                                   (f_i[i] - self.nvalence)) /
                                  (f_i[i] - f_i[i - 1]))

        # XXX broadcast would be better!
        return wfs.kptband_comm.sum(fermilevel)

    def find_fermi_level(self, wfs, ne, fermilevel, spins={0, 1, None}):
        niter = 0

        x = self.fermilevel
        if not np.isfinite(x):
            x = self.guess_fermi_level(wfs)

        data = np.empty(4)

        def f(x, data=data):
            data[:] = 0.0
            for kpt in wfs.kpt_u:
                if kpt.s in spins:
                    data += self.distribution(kpt, x)
            wfs.kptband_comm.sum(data)
            n, dnde = data[:2]
            dn = n - ne
            return dn, dnde

        fermilevel, niter = findroot(f, x)

        self.niter = niter
        magmom, e_entropy = data[2:]
        return fermilevel, magmom, e_entropy


class FermiDirac(SmoothDistribution):
    def __init__(self, width, fixmagmom=False):
        SmoothDistribution.__init__(self, width, fixmagmom)

    def todict(self):
        dct = SmoothDistribution.todict(self)
        dct['name'] = 'fermi-dirac'
        return dct

    def __str__(self):
        s = '  Fermi-Dirac: width={0:.4f} eV\n'.format(self.width * Hartree)
        return SmoothDistribution.__str__(self) + s

    def distribution(self, kpt, fermilevel):
        x = (kpt.eps_n - fermilevel) / self.width
        x = x.clip(-100, 100)
        y = np.exp(x)
        z = y + 1.0
        kpt.f_n[:] = kpt.weight / z
        n = kpt.f_n.sum()
        dnde = (n - (kpt.f_n**2).sum() / kpt.weight) / self.width
        y *= x
        y /= z
        y -= np.log(z)
        e_entropy = kpt.weight * y.sum() * self.width
        sign = 1 - kpt.s * 2 if kpt.s is not None else 0.0
        return np.array([n, dnde, n * sign, e_entropy])

    def extrapolate_energy_to_zero_width(self, E):
        return E - 0.5 * self.e_entropy


class MethfesselPaxton(SmoothDistribution):
    def __init__(self, width, order=0, fixmagmom=False):
        SmoothDistribution.__init__(self, width, fixmagmom)
        self.order = order

    def todict(self):
        dct = SmoothDistribution.todict(self)
        dct['name'] = 'methfessel-paxton'
        dct['order'] = self.order
        return dct

    def __str__(self):
        s = '  Methfessel-Paxton: width={0:.4f} eV, order={1}\n'.format(
            self.width * Hartree, self.order)
        return SmoothDistribution.__str__(self) + s

    def distribution(self, kpt, fermilevel):
        x = (kpt.eps_n - fermilevel) / self.width
        x = x.clip(-100, 100)

        z = 0.5 * (1 - erf(x))
        for i in range(self.order):
            z += (self.coff_function(i + 1) *
                  self.hermite_poly(2 * i + 1, x) * np.exp(-x**2))
        kpt.f_n[:] = kpt.weight * z
        n = kpt.f_n.sum()

        dnde = 1 / np.sqrt(pi) * np.exp(-x**2)
        for i in range(self.order):
            dnde += (self.coff_function(i + 1) *
                     self.hermite_poly(2 * i + 2, x) * np.exp(-x**2))
        dnde = dnde.sum()
        dnde *= kpt.weight / self.width
        e_entropy = (0.5 * self.coff_function(self.order) *
                     self.hermite_poly(2 * self.order, x) * np.exp(-x**2))
        e_entropy = -kpt.weight * e_entropy.sum() * self.width

        sign = 1 - kpt.s * 2
        return np.array([n, dnde, n * sign, e_entropy])

    def coff_function(self, n):
        return (-1)**n / (np.product(np.arange(1, n + 1)) *
                          4**n * np.sqrt(np.pi))

    def hermite_poly(self, n, x):
        if n == 0:
            return 1
        elif n == 1:
            return 2 * x
        else:
            return (2 * x * self.hermite_poly(n - 1, x) -
                    2 * (n - 1) * self.hermite_poly(n - 2, x))

    def extrapolate_energy_to_zero_width(self, E):
        return E - self.e_entropy / (self.order + 2)


class FixedOccupations(ZeroKelvin):
    def __init__(self, occupation):
        self.occupation = np.array(occupation)
        ZeroKelvin.__init__(self, True)

    def spin_paired(self, wfs):
        return self.fixed_moment(wfs)

    def fixed_moment(self, wfs):
        for kpt in wfs.kpt_u:
            wfs.bd.distribute(self.occupation[kpt.s], kpt.f_n)


class TFOccupations(FermiDirac):
    def __init__(self):
        FermiDirac.__init__(self, width=0.0, fixmagmom=False)

    def todict(self):
        return {'name': 'orbital-free'}

    def occupy(self, f_n, eps_n, ne, weight=1):
        """Fill in occupation numbers.

        In TF mode only one band. Is guaranteed to work only
        for spin-paired case.

        return HOMO and LUMO energies."""
        # Same as occupy in FermiDirac expect one band: weight = ne
        return FermiDirac.occupy(self, f_n, eps_n, ne, ne)
