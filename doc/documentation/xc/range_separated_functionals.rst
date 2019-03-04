.. _rsf:

=================================
Range separated functionals (RSF)
=================================

Introduction
============

Range separated functionals (RSF) are a subgroup of hybrid
functionals. While conventional (global) hybrid functionals like PBE0
or B3LYP use fixed fractions of Hartree-Fock (HFT, E\ :sub:`XX`\ )
and DFT (E\ :sub:`X`\ )
exchange for exchange, f.e. 1/4 E\ :sub:`XX`\  and 3/4 E\ :sub:`X`\  in
the case of PBE0,
RSFs mix the two contributions by the spatial distance between two points, `r_{12}`,
using a soft function `\omega_\mathrm{RSF}(\gamma, r_{12})`.

To achieve this, the coulomb interaction kernel,
`\frac{1}{r_{12}} = \frac{1}{|r_1 - r_2|}`,
which appears in the exchange integral from HFT is split into two parts:

`\frac{1}{r_{12}} = \underbrace{\frac{1 - [\alpha + \beta ( 1 - \omega_\mathrm{RSF} (\gamma, r_{12}))]}{r_{12}}}_{\text{SR, DFT}} + \underbrace{\frac{\alpha + \beta ( 1 - \omega_\mathrm{RSF} (\gamma, r_{12}))}{r_{12}}}_{\text{LR, HFT}}`,

the short-range (SR) part is handled by the exchange from a (semi-)local LDA
or GGA functional such as PBE, while the long-range part (LR) is handled by
the exchange from HFT. `\alpha` and `\beta` are functional dependent mixing
parameters.  `\alpha \ne 0` and `\beta = 0` resembles conventional global
hybrids. RSFs with `\alpha = 0` and `\beta \ne 0` are usually denoted by
``LC`` and the name of the semi-local functional, f.e. LC-PBE.
RSFs with `\alpha \ne 0` and `\beta \ne 0` are usually denoted by
``CAM`` and the name of the semi-local functional, f.e. CAM-BLYP.

For the separating function `\omega_\mathrm{RSF}`, two functions are in common
use: either the complementary error function,
`\omega_\mathrm{RSF} = \mathrm{erfc}(\gamma r_{12})`, or the Slater-function,
`\omega_\mathrm{RSF} = e^{(-\gamma r_{12})}`. While the use of the
complementary error function is computationally fortunate for codes utilizing
Gaussian type basis sets, the Slater-function give superior results in the
calculation of Rydberg-state and charge transfer excitations. To distinguish
between these both functions, functionals using the Slater-function append the letter "Y" to the RSF marker, f.e. LCY-PBE or CAMY-B3LYP, while functionals
using the complementary error function keep the marker as it is, f.e.
LC-PBE or CAM-B3LYP.

Besides `r_{12}`, the separation functions use a second parameter, the
screening factor `\gamma`. The optional value for `\gamma` is under
discussion. A density dependence is stated. For most RSF standard values
for `\gamma` are defined, although it is possible to tune `\gamma` to optimal
values for calculations investigating ionization potentials, charge transfer
excitations and the binding curves of bi-radical cations.

Implementation
==============

The implementation of RSFs in gpaw consists of two parts:
 * once the implementation of the semi-local functional part. This is
   done in libxc.
 * once the implementation of the Hartree-Fock exchange. This is done
   in ``hybrid.py``.

As range separating function the Slater-function,
`\omega_\mathrm{RSF} = e^{(-\gamma r_{12})}`,
is used. Besides the possibility to set `\gamma` to an arbitrary
value, the following functionals were implemented:

========== ======== ======= ===================== =========
Functional `\alpha` `\beta` `\gamma` (`a_0^{-1}`) Reference
========== ======== ======= ===================== =========
CAMY-BLYP  0.2      0.8     0.44                  [AT08]_
CAMY-B3LYP 0.19     0.46    0.34                  [SZ12]_
LCY-BLYP   0.0      1.0     0.75                  [SZ12]_
LCY-PBE    0.0      1.0     0.75                  [SZ12]_
========== ======== ======= ===================== =========

As the implementation of RSFs in gpaw is based on the finite difference
exact exchange code (hybrid.py), the implementation inherits its positive
and negative properties, in summary:

* self-consistent calculations using RSFs
* calculations can only be done for the `\Gamma` point
* only non-periodic boundary conditions can be used
* only RMMDIIS can be used as eigensolver

Important: As one of the major benefits of the RSF is to retain the
`\frac{1}{r}` asymptote of the exchange potential, one has to use
large boxes if neutral or anionic systems are considered. Large boxes
start at 6Å vacuum around each atom. For anionic systems "large" should
be extended.

Further information about the implementation and RSFs can be found in
[WW18]_ and in detail in [Wu16]_.

Simple usage
============

In general calculations using RSF can simply be done choosing the appropriate
functional as in the following snippet:

.. literalinclude:: rsf_simple.py

Three main points can be seen already in this small snippet. Even if choosing
the RSF is quite simple by choosing ``xc=LCY-PBE``, one has to choose RMMDIIS
as eigensolver, ``eigensolver=RMMDIIS()``, and has to decrease the
convergence criteria a little.

Improving results
=================

However, there are a few drawbacks, at first in an SCF calculation the
contributions from the core electrons are also needed, which have to be
calculated during the generation of the PAW datasets. Second: for the
calculation of the exchange on the Cartesian grid, the (screened) Poisson
equation has to be solved numerically. For a charged system, as f.e. the
exchange of a state with itself, one has to neutralize the charge by
subtracting a Gaussian representing the "over-charge", solve the
(screened) Poisson-equation for the neutral system and add the solution
for the Gaussian to the solution for the neutral system. However, if the
charge to remove is "off-center", the center of the neutralizing charge
should match the center of the "over-charge"
preventing an artificial dipole. The latter is done by using a Poisson solver
which uses the charge center for removal:
``poissonsolver=PoissonSolver(use_charge_center=True)``.
The next listing shows these two steps:


.. literalinclude:: rsf_setup_poisson.py

The generation of PAW-datasets can also be done by
``gpaw-setup -f PBE -x --gamma=0.75 C O``

Tuning `\gamma`
===============

As stated in the introduction, the optimal value for `\gamma` is under
discussion. One way to find the optimal value for `\gamma` for ionization
potentials is to tune `\gamma` in a way, that the negative eigenvalue of the
HOMO matches the calculated IP. To use different values of `\gamma`, one has
to pass the desired value of `\gamma` to the variable ``omega``.

.. literalinclude:: rsf_gamma.py

linear response TDDFT
=====================

One of the major benefits of RSF is their ability to describe long-range charge transfer by
linear response time-dependent DFT (lrTDDFT). If one uses RSF with lrTDDFT one has at least
to activate the use of the Fock operator (FO) on the unoccupied states. Also the charge
centered compensation of the over charge should be activated, see [Wu16]_ for details.
The use of the FO on the unoccupied states is activated by the keyword ``unocc=True`` as in 
the following code:

.. literalinclude:: rsf_lrtddft.py


.. [AT08] Y. Akinaga and S. Ten-no. *Range-separation by the Yukawa potential in long-range corrected density functional theory with Gaussian-type basis functions*. Chemical Physics Letters 462.4 (10. Sep. 2008), S. 348–351.

.. [SZ12] M. Seth and T. Ziegler. *Range-Separated Exchange Functionals with Slater-Type Functions*. J. Chem. Theory Comput. 8.3 (2012), S. 901–907.

.. [Wu16] R. Würdemann. *Berechnung optischer Spektren und Grundzustandseigenschaften neutraler und geladener Moleküle mittels Dichtefunktionaltheorie*, PhD-Thesis. DOI: 10.6094/UNIFR/11315

.. [WW18] R. Würdemann and M. Walter. *Charge Transfer Excitations with Range Separated Functionals Using Improved Virtual Orbitals*. J. Chem. Theory Comput. 14.7 (2018), S. 3667-3676
