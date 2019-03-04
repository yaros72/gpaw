.. _gw_theory:

=======================================================
Quasi-particle spectrum in the GW approximation: theory
=======================================================

The foundations of the GW method are described in Refs. \ [#Hedin1965]_ and \
[#Hybertsen1986]_. The implementation in GPAW is documented in Ref. \
[#Hueser2013]_.

For examples, see :ref:`gw tutorial`.


Introduction
============

Quasi-particle energies are obtained by replacing the DFT exchange-
correlation contributions by the GW self energy and exact exchange:

.. math:: E_{n \mathbf{k}} = \epsilon_{n \mathbf{k}} + Z_{n \mathbf{k}} \cdot \text{Re} \left(\Sigma_{n \mathbf{k}}^{\vphantom{\text{XC}}} + \epsilon^{\text{EXX}}_{n \mathbf{k}} - V^{\text{XC}}_{n \mathbf{k}} \right)

where `n` and `\mathbf{k}` are band and k-point indices, respectively.

The different contributions are:

`\epsilon_{n \mathbf{k}}`: Kohn-Sham eigenvalues taken from a groundstate
calculation

`V^{\text{XC}}_{n \mathbf{k}}`: DFT exchange-correlation contributions
extracted from a groundstate calculation

`\epsilon^{\text{EXX}}_{n \mathbf{k}}`: exact exchange contributions

The renormalization factor is given by:

.. math:: Z_{n \mathbf{k}} = \left(1 - \text{Re}\left< n \mathbf{k}\middle| \frac{\partial}{\partial\omega} \Sigma(\omega)_{|\omega = \epsilon_{n \mathbf{k}}}\middle| n \mathbf{k}\right>\right)^{-1}

`\left| n \mathbf{k} \right>` denotes the Kohn-Sham wavefunction which is
taken from the groundstate calculation.

The self energy is expanded in plane waves, denoted by `\mathbf{G}` and
`\mathbf{G}'`:

.. math:: \Sigma_{n \mathbf{k}}(\omega = \epsilon_{n \mathbf{k}}) =& \left<n \mathbf{k} \middle| \Sigma(\omega) \middle|n \mathbf{k} \right>\\
 =& \frac{1}{\Omega} \sum\limits_{\mathbf{G} \mathbf{G}'} \sum\limits_{\vphantom{\mathbf{G}}\mathbf{q}}^{1. \text{BZ}} \sum\limits_{\vphantom{\mathbf{G}}m}^{\text{all}} \frac{i}{2 \pi} \int\limits_{-\infty}^\infty\!d\omega'\, W_{\mathbf{G} \mathbf{G}'}(\mathbf{q}, \omega') \, \cdot \\
 & \frac{\rho^{n \mathbf{k}}_{m \mathbf{k} - \mathbf{q}}(\mathbf{G}) \rho^{n \mathbf{k}*}_{m \mathbf{k} - \mathbf{q}}(\mathbf{G}')}{\omega + \omega' - \epsilon_{m \, \mathbf{k} - \mathbf{q}} + i \eta \, \text{sgn}(\epsilon_{m \, \mathbf{k} - \mathbf{q}} - \mu)}

where `m` runs both over occupied and unoccupied bands and `\mathbf{q}`
covers the differences between all k-points in the first Brillouin zone.
`\Omega = \Omega_\text{cell} \cdot N_\mathbf{k}` is the volume and `\eta` an
(artificial) broadening parameter. `\mu` is the chemical potential.

The screened potential is calculated from the (time-ordered) dielectric
matrix in the Random Phase Approximation:

.. math:: W_{\mathbf{G} \mathbf{G}'}(\mathbf{q}, \omega) = \frac{4 \pi}{|\mathbf{q} + \mathbf{G}|} \left( (\varepsilon^{\text{RPA}-1}_{\mathbf{G} \mathbf{G}'}(\mathbf{q}, \omega) - \delta^{\vphantom{\text{RPA}}}_{\mathbf{G} \mathbf{G}'} \right) \frac{1}{|\mathbf{q} + \mathbf{G}'|}

Refer to :ref:`df_theory` for details on how the response function and the
pair density matrix elements `\rho^{n \mathbf{k}}_{m \mathbf{k} -
\mathbf{q}}(\mathbf{G}) \equiv \left<n \mathbf{k} \middle| e^{i(\mathbf{q} +
\mathbf{G})\mathbf{r}} \middle|m \, \mathbf{k} \!-\! \mathbf{q} \right>`
including the PAW corrections are calculated.


Coulomb divergence
==================


The head of the screened potential (`\mathbf{G} = \mathbf{G}' = 0`) diverges
as `1/q^2` for `\mathbf{q} \rightarrow 0`. This divergence, however, is
removed for an infinitesimally fine k-point sampling, as
`\sum\limits_{\mathbf{q}} \rightarrow \frac{\Omega}{(2\pi)^3} \int\!d^3
\mathbf{q} \propto q^2`. Therefore, the `\mathbf{q} = 0` term can be
evaluated analytically, which yields:

.. math:: W_{\mathbf{00}}(\mathbf{q}=0, \omega) = \frac{2\Omega}{\pi} \left(\frac{6\pi^2}{\Omega}\right)^{1/3} \varepsilon^{-1}_{\mathbf{00}}(\mathbf{q} \rightarrow 0, \omega)

for the head and similarly

.. math:: W_{\mathbf{G0}}(\mathbf{q}=0, \omega) = \frac{1}{|\mathbf{G}|} \frac{\Omega}{\pi} \left(\frac{6\pi^2}{\Omega}\right)^{2/3} \varepsilon^{-1}_{\mathbf{G0}}(\mathbf{q} \rightarrow 0, \omega)

for the wings of the screened potential. Here, the dielectric function is
used in the optical limit.

This is only relevant for the terms with `n = m`, as otherwise the pair
density matrix elements vanish: `\rho^{n \mathbf{k}}_{m \mathbf{k}} = 0` for
`n \neq m`.


Frequency integration
=====================

`\rightarrow` ``domega0, omega2``


The frequency integration is performed numerically on a user-defined grid for
positive values only. This is done by rewriting the integral as:

.. math:: & \int\limits_{-\infty}^\infty\!d\omega'\, \frac{W(\omega')}{\omega + \omega' - \epsilon_{m \, \mathbf{k} - \mathbf{q}} \pm i \eta}\\
 =& \int\limits_{0}^\infty\!d\omega'\, W(\omega') \left(\frac{1}{\omega + \omega' - \epsilon_{m \, \mathbf{k} - \mathbf{q}} \pm i \eta} + \frac{1}{\omega - \omega' - \epsilon_{m \, \mathbf{k} - \mathbf{q}} \pm i \eta}\right)

with the use of `W(\omega') = W(-\omega')`.

The frequency grid is the same as that used for the dielectric function. Read more about it here: :ref:`df_tutorial_freq`.



.. _gw_theory_ppa:

Plasmon Pole Approximation
==========================

`\rightarrow` ``ppa = True``


Within the plasmon pole approximation (PPA), the dielectric function is
modelled as a single peak at the main plasmon frequency
`\tilde{\omega}_{\mathbf{G}\mathbf{G}'}(\mathbf{q})`:

.. math:: \varepsilon^{-1}_{\mathbf{G}\mathbf{G}'}(\mathbf{q}, \omega) = R _{\mathbf{G}\mathbf{G}'}(\mathbf{q}) \left(\frac{1}{\omega - \tilde{\omega}_{\mathbf{G}\mathbf{G}'}(\mathbf{q}) + i\eta} - \frac{1}{\omega + \tilde{\omega}_{\mathbf{G}\mathbf{G}'}(\mathbf{q}) - i\eta}\right)

The two parameters are found by fitting this expression to the full
dielectric function for the values `\omega = 0` and `\omega = i E_0`:

.. math:: \varepsilon^{-1}_{\mathbf{G}\mathbf{G}'}(\mathbf{q}, 0) =& \frac{-2 R}{\tilde{\omega}} \hspace{0.5cm} \varepsilon^{-1}_{\mathbf{G}\mathbf{G}'}(\mathbf{q}, iE_0) = \frac{-2 R \tilde{\omega}}{E_0^2 + \tilde{\omega}^2}\\
 \Rightarrow \tilde{\omega}_{\mathbf{G}\mathbf{G}'}(\mathbf{q}) =& E_0 \sqrt{\frac{\varepsilon^{-1}_{\mathbf{G}\mathbf{G}'}(\mathbf{q}, iE_0)} {\varepsilon^{-1}_{\mathbf{G}\mathbf{G}'}(\mathbf{q}, 0) - \varepsilon^{-1}_{\mathbf{G}\mathbf{G}'}(\mathbf{q}, iE_0)}}\\
 R _{\mathbf{G}\mathbf{G}'}(\mathbf{q}) =& -\frac {\tilde{\omega}_{\mathbf{G}\mathbf{G}'}(\mathbf{q})}{2} \varepsilon^{-1}_{\mathbf{G}\mathbf{G}'}(\mathbf{q}, 0)

In this way, the frequency integration for the self energy can be evaluated
analytically. The fitting value `E_0` has to be chosen carefully. By default,
it is 1 H.


Hilbert transform
=================

The self-energy is evaluated using the Hilbert transform technique described in \ [#Kresse2006]_ .


Parallelization
===============

`\rightarrow` ``nblocks = int``


By default, the calculation is fully parallelized over k-points and bands. If more memory is required for storing the response function in the plane wave basis, additional block parallelization is possible. This distributes the matrix amongst the number of CPUs specified by ``nblocks``, resulting in a lower total memory requirement of the node. ``nblocks`` needs to be an integer divisor of the number of requested CPUs.


I/O
===


All necessary informations of the system are read from ``calc =
'filename.gpw'`` which must contain the wavefunctions. This is done by
performing ``calc.write('groundstate.gpw', 'all')`` after the groundstate
calculation. GW supports spin-paired planewave calculations.

The exchange-correlation contribution to the Kohn-Sham eigenvalues is stored in ``'filename.vxc.npy'`` and the exact-exchange eigenvalues are stored in ``'filename.exx.npy'``.
The resulting output is written to ``'filename_results.pckl'`` and a summary of input as well as a output parameters are given in the human-readable  ``'filename.txt'`` file. Information about the calculation of the screened coulomb interaction is printed in ``'filename.w.txt'``.


Convergence
===========

The results must be converged with respect to:

- the number of k-points from the groundstate calculation

    A much finer k-point sampling might be required for converging the GW
    results than for the DFT bandstructure.

- the number of bands included in the calculation of the self energy ``nbands``

- the planewave energy cutoff ``ecut``
    
    ``ecut`` and ``nbands`` do not converge independently. As a rough
    estimation, ``ecut`` should be around the energy of the highest included
    band. If ``nbands`` is not specified it will be set equal to the amount of plane waves determined by ``ecut``.

- the number of frequency points ``domega0, omega2``

    The grid needs to resolve the features of the DFT spectrum.

- the broadening ``eta``

    This parameter is only used for the response function and in the plasmon
    pole approximation. Otherwise, it is automatically set to `\eta = 0.1`.


Parameters
==========
For input parameters, see :ref:`gw tutorial`.


References
==========


.. [#Hedin1965] L. Hedin,
                "New Method for Calculating the One-Particle Green's Function with Application to the Electron-Gas Problem",
                *Phys. Rev.* **139**, A796 (1965).

.. [#Hybertsen1986] M.S. Hybertsen and S.G. Louie,
                    "Electron correlation in semiconductors and insulators: Band gaps and quasiparticle energies",
                    *Phys. Rev. B* **34**, 5390 (1986).

.. [#Hueser2013] F. Hüser, T. Olsen, and K. S. Thygesen,
                 "Quasiparticle GW calculations for solids, molecules, and two-dimensional materials",
                 *Phys. Rev. B* **87**, 235132 (2013).

.. [#Kresse2006] M. Shishkin and G. Kresse,
                 "Implementation and performance of the frequency-dependent GW method within the PAW framework",
                 *Phys. Rev. B* **74**, 035101 (2006).
