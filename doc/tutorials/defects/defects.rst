.. _defects_tutorial:

=====================================================
Calculating the formation energies of charged defects
=====================================================


Introduction
============

The energy required to form a point defect in an otherwise
pristine sample is usually calculated using the so-called
Zhang-Northrup formula [#RMP]_:

.. math::

  E^f[X^q] = E[X^q] - E_0 - \sum_i\mu_in_i + q (\epsilon_v + \epsilon_F)

In this formula, `X` labels the type of defect (e.g. a gallium vacancy
`\mathrm{V_{Ga}}` or zinc interstitial `\mathrm{Zn_i}`) and `q` its charge
state, i.e. the net charge contained in some volume surrounding the defect.
`q` is defined such that `q=-1` for an electron.  `E[X^q]` is the total
energy of the sample with the defect, and `E_0` the energy of the pristine
(bulklike) sample. In general, we form the defect by changing the number of
(neutral) atoms of species `i` in the sample by `n_i`; the change in energy
due to this addition (or removal) of atoms is quantified by the chemical
potentials `\mu_i` of each species. Similarly, if we require the addition or
removal of electrons to form the defect (i.e. to obtain a nonzero charge
state) we also change the energy according to the chemical potential of the
electrons as `N_e \mu_e`.  `N_e` is simply related to `q` as `N_e = -q`,
while conventionally `\mu_e` is written in terms of a "Fermi Energy"
referenced to the valence band maximum `\epsilon_v`, i.e. `\mu_e = \epsilon_v
+ \epsilon_F`. Taking everything together, the formation energy can thus be
interpreted as the difference in total energy of the sample containing a
defect and the total energy of all the constituents required to form the
defect (the pristine sample, atoms and electrons).  In general one expects a
defect formation energy to be positive, so that it costs energy to make a
defect. The formation energy will also depend on the chemical potentials of
the atoms and of the electrons, reflecting the growth conditions of the
sample.

Within periodic boundary conditions, the quantity `E[X^q] - E_0` is obtained by
constructing supercells of the pristine unit cell, and then calculating the
difference in total energies of the supercells with and without the defect. In
the limit of an infinitely large supercell, the dilute limit of a single,
isolated defect should be achieved. In practice however, a variety of finite
size effects can lead to slow convergence with supercell size [#Lany]_. In the
case of nonzero charge states, the electrostatic interaction between the
periodically-repeated array of defects leads to particularly slow convergence.
In this tutorial, we apply the method proposed by Freysoldt, Neugebauer and Van
de Walle (FNV) [#FNV]_ to correct for this effect in a bulk system, namely the
triply-charged gallium vacancy in GaAs, as well as in a 2D system.


Theoretical background: The FNV scheme
======================================

Here we outline the FNV approach to correcting for the electrostatic
interactions; more details can be found in [#FNV]_.  A practical example is
given in the next section. The electrostatic energy of a periodically-
repeated charged system is divergent.  Therefore, the calculation of `E[X^q]`
in periodic boundary conditions is only possible if one adds a homogeneous
neutralising background charge of density `-q/\Omega` (where `\Omega` is the
volume of the supercell). By taking the limit `\Omega \rightarrow \infty`
interactions originating from copies of the charge distribution and from the
background charge are removed. The FNV scheme aims to accelerate this
convergence by employing the following correction:

.. math::

  E[X^q] - E_0  =
  (E[X^q] - E_0)_\mathrm{uncorrected} - E_{\mathrm{l}} + q\Delta V

The uncorrected term in brackets is the total energy difference one obtains
from calculations employing periodic boundary conditions, which include the
background charge. The first correction term, the *lattice term*
`E_{\mathrm{l}}`, is the electrostatic energy per unit cell of a
periodically-repeated array of model charges immersed in the neutralising
background, minus the interaction of the model charge with itself. The second
correction term, the *alignment term* `q\Delta V`, ensures that the zero
point (d.c. component) of the electrostatic potential of the calculation with
the defect is consistent with that used when determine the valence band edge
`\epsilon_v`. In practice this is achieved by choosing `\Delta V` to align
the electrostatic potential of the defect-containing supercell--- in a region
of space located far from the defect itself--- with the electrostatic
potential of a pristine supercell.

First we consider the lattice term.
The key idea of the FNV scheme is to introduce a model charge distribution
which is designed to simulate the actual distribution of charge around
a defect.  A simple choice of model is a 3D gaussian:

.. math::

  \rho^m(r) = \frac{q}{[\sqrt{2\pi}\sigma]^3} e^{-r^2/(2\sigma^2)}

which integrates to `q` and has a full-width at half maximum (FWHM)
of `2\sigma \sqrt{2 \ln 2}`.
The width is a parameter of the model but should somewhat reflect the
real defect charge distribution obtained as the difference between
bulk and defect calculations, `\rho^{X^q}(\vec{r}) - \rho^0(\vec{r})`.
In principle more exotic model distributions can be used, e.g. a combination
of a gaussian and an exponential [#Komsa]_ .

The calculation of `E_{\mathrm{l}}` is most conveniently done in Fourier space.
Within a linear, isotropic and homogeneous dielectric characterised
by `\varepsilon`, `\rho^m` generates
an electrostatic potential given by

.. math::
  V(\vec{G}\neq0) = \frac{4\pi}{\varepsilon G^2} \rho^m(\vec{G})

where the `\vec{G}`'s are reciprocal lattice vectors. `E_{\mathrm{l}}` is then
obtained as

.. math::
  E_{\mathrm{l}} = \frac{2\pi} {\varepsilon \Omega} \sum_{\vec{G} \neq 0}
  \frac{|\rho^m(\vec{G})|^2}{G^2} -
  \frac{1}{\pi\varepsilon} \int_0^{\infty} \mathrm{d}g
  |\rho^{m}(g)|^2.


The first term is the energy of all the periodic repeats of `\rho^m`; the
inclusion of the neutralising background means the `\vec{G}=0` term is
omitted. The second term is the electrostatic energy of `\rho^m` interacting
with itself, where here we have implicitly assumed that `\rho^m`  is
spherically symmetric.

For the case of the gaussian,

.. math::
  \rho^m(G) = q e^{-G^2\sigma^2/2}

so

.. math::
  E_{\mathrm{l}} = \frac{2\pi} {\varepsilon \Omega} \sum_{\vec{G} \neq 0}
  \frac{q^2 e^{-G^2\sigma^2}}{G^2}
  -
  \frac{q^2}{2\sqrt{\pi}\varepsilon\sigma}.

Now we turn to the alignment term. As stated above, `\Delta V` applies a
constant shift to the electrostatic potential of the supercell containing the
defect, `V^{X^q}_\mathrm{el}` such that a point `\vec{r_0}` located far from
the defect, the potential is bulklike, i.e.\ `V^{0}_\mathrm{el}`. In
principle this means applying a shift

.. math::
  \Delta V'(\vec{r_0}) = V^{0}_\mathrm{el}(\vec{r_0}) -
  V^{X^q}_\mathrm{el}(\vec{r_0})

However, the problem is that the defect has a long-range effect on the
electrostatic potential, such that even if `\vec{r_0}` is located many
angstroms away from the defect, the potential
`V^{X^q}_\mathrm{el}(\vec{r_0})` is not truly bulklike. The FNV solution is
to suppose that the potential due to the model charge, `V(\vec{r})`,
accurately describes the long-range behaviour of the true defect charge
distribution, so that its effects can be removed from `V^{X^q}_\mathrm{el}`
by a simple subtraction. Thus we introduce

.. math::
  \Delta V(\vec{r}) = V^{0}_\mathrm{el}(\vec{r}) - [V^{X^q}_\mathrm{el}(\vec{r}) - V(\vec{r})]
                    = V(\vec{r}) - [V^{X^q}_\mathrm{el}(\vec{r}) - V^{0}_\mathrm{el}(\vec{r}) ]

where `V(\vec{r})` just the Fourier transform of `V(\vec{G})` above.

A remaining problem is that `\Delta V(\vec{r})` is a strongly varying
function of space, so we cannot simply set `\Delta V = \Delta V(\vec{r_0})`.
Instead, some spatial averaging scheme is required.  One option is to perform
a planar average, for instance in the `xy` plane of area `A`:

.. math::
  \Delta V(z) = \frac{1}{A} \int_{A} \mathrm{d}x \mathrm{d}y \Delta V(\vec{r})

`\Delta V` should then be taken as `\Delta V(z_0)`, where `z_0` is the plane
furthest from the defect. An alternative option is to perform the average
over some volume `\tau` centred on each atom `J`, i.e.

.. math::

  \Delta V(J) = \frac{1}{\tau} \int_{\tau_J} \mathrm{d}\vec{r} V^{0}_\mathrm{el}(\vec{r}) - \frac{1}{\tau} \int_{\tau_J} \mathrm{d}\vec{r} [V^{X^q}_\mathrm{el}(\vec{r}) - V(\vec{r})]

The reason for partitioning the equation as above is that if one performs a
full relaxation in the presence of a defect, even bulklike atoms may undergo
some change in position. The above averaging takes this into account by
allowing the averaging volume `\tau_J` to track the position of the atom.
Using this scheme `\Delta V` should then be taken as `\Delta V(J_0)`, where
`J_0` labels an atom far from the defect.


The Ga vacancy in GaAs
======================

We now apply the FNV scheme to the triple-negatively charged (`q = -3`) Ga
vacancy in GaAs, which a system also considered in Ref. [#FNV]_. Due to the
high charge state of the defect, electrostatic effects are particularly
important here. We here consider an NxNxN supercell of GaAs, which contains
8*N**3 atoms. The script below calculates the total energies of the supercell
with and without the defect, where we created the vacancy by removing the Ga
atom at (0,0,0). Note how we set the charge in the defect calculation, and
that we save the gpw files for further processing. Also, note that we do not
perform a relaxation for the system with the defect. For N=2 this script
takes around 30 minutes to complete using 8 processors.

.. literalinclude:: gaas.py

By subtracting the energy of the pristine system from the energy of the
defective system, we obtain an uncorrected total energy difference `(E[X^q] -
E_0)_\mathrm{uncorrected}` of 21.78 eV.

We now calculate the FNV corrections. Here we take a dielectric constant of
12.7 which is the clamped-ion static limit (i.e. the low frequency dielectric
constant excluding the effects of ionic relaxation). We use a Gaussian model
charge centred at (0, 0, 0) with a standard deviation of 0.72 Bohr
(corresponding to a FWHM of 2 Bohr).

The script ``electrostatics.py`` takes the gpw files of the defective and
pristine calculation as input, as well as the gaussian parameters and
dielectric constant, and calculates the different terms in the correction
scheme. For this case, the calculated value of `E_{\mathrm{l}}` is -1.28 eV.

.. literalinclude:: electrostatics.py

The script also produces an output file ``electrostatic_data.npz`` which
gives the function `\Delta V(z)` introduced above, and also the planar
averages of the model potential and the difference between the planar
averages of the defective and pristine electrostatic potentials. We can plot
the data using the following script

.. literalinclude:: plot_potentials.py

This gives the following plot:

.. image:: planaraverages.png
           :scale: 50 %

According to the recipe introduced above, we extract the constant `\Delta V`
from `\Delta V(z)` furthest from the defect, corresponding to the middle of
the unit cell. The extracted value of `\Delta V = -0.14` eV is shown as the
dashed line. Note that such a plot provides a consistency check of the FNV
scheme; if `\Delta V(z)` does not display flat behaviour away from the
defect, it is a sign that the model is not describing the true electrostatics
sufficiently well.

Taken together, the corrected energy difference is

.. math::

  E[V_\mathrm{Ga}^{-3}] - E_0  = [21.78  - (-1.28 ) + (-3)\times(-0.14)] \mathrm{eV}
                               = [21.78 + 1.70] \mathrm{eV}
                               = [23.5] \mathrm{eV}

This case is a rather extreme example, since the supercell is rather small and
the charge state is large. Nonetheless, the large correction demonstrates the
importance of electrostatics.

The above calculation can be repeated for different sizes of supercells. The
plot below shows the energy differences before and after the FNV corrections
are applied, as a function of the number of atoms in the supercell (c.f. Fig.
5 of [#FNV]_). The corrections nicely remove the slow convergence due to the
electrostatics. Note also that even for the largest supercell (4x4x4, 512
atoms), the electrostatic correction is still large, 0.7 eV.

.. image:: energies.png
           :scale: 50 %


Additional remarks on calculating formation energies
====================================================

Here we briefly discuss the other ingredients needed to calculate defect
formation energies using the Zhang-Northrup formula. First, the valence band
position `\epsilon_v` must be obtained from a calculation on the pristine unit
cell, with a dense enough `k`-point sampling so that the band edge is included
(for GaAs this just means that the `\Gamma` point is included). Because GPAW
always sets the average electrostatic potential to zero, this value is already
aligned to the supercell calculation of the pristine sample so needs no further
adjustment.

The chemical potentials `\mu_i` can be varied, but only within certain limits.
For the gallium vacancy we require a value of `\mu_\mathrm{Ga} \equiv
\mu_\mathrm{Ga}[\mathrm{GaAs}]` , which lies within the range [#RMP]_:

.. math::

 \mu_\mathrm{Ga}[\mathrm{bulk \ Ga}] > \mu_\mathrm{Ga}[\mathrm{GaAs}]>
 \mu_\mathrm{Ga}[\mathrm{bulk \ Ga}] + \Delta H_\mathrm{f}[\mathrm{GaAs}]

Here, `\Delta H_\mathrm{f}` is the enthalpy of formation, and
`\mu_\mathrm{Ga}[\mathrm{bulk \ Ga}]` the chemical potential corresponding to
equilibrium with bulk gallium. Normally one would consider the two limits
`\mu_\mathrm{Ga}[\mathrm{GaAs}] = \mu_\mathrm{Ga}[\mathrm{bulk \ Ga}]` ("Ga
rich") and `\mu_\mathrm{Ga}[\mathrm{GaAs}] = \mu_\mathrm{Ga}[\mathrm{bulk \
Ga}] + \Delta H_\mathrm{f}[\mathrm{GaAs}]` ("As rich", or "Ga poor").
`\mu_\mathrm{Ga}[\mathrm{bulk \ Ga}]` and `\Delta
H_\mathrm{f}[\mathrm{GaAs}]` can be obtained from total energy calculations
on bulk Ga, As, and GaAs.

Carrying out the necessary calculations yields values of 4.75 eV for
`\epsilon_v` and -3.59 eV for `\mu_\mathrm{Ga}[\mathrm{bulk \ Ga}]`. Hence the
defect formation energy calculated for the 222 cell with and without the FNV
corrections, assuming Ga rich conditions, is 5.6 and 3.9 eV respectively (here
we also set the position of the electron chemical potential to the top of the
valence band, i.e. `\epsilon_F` = 0.


Calculation electrostatic corrections in two dimensions
=======================================================

For two-dimensional systems, the approach described above is complicated by the
extreme anisotropy of the system, and in particular by the very weak screening
occurring between periodically repeated layers. This means that the
electrostatic interaction between a localised charge state and its images in
neighbouring cells is generally strong, and the corresponding correction is
large. The strong interactions pose a number of problems for the correction
scheme described above: The heart of the correction scheme is to write the
formation energy of the infinitely dilute system as

.. math::
  E^f[X^q]_{\infty} = E^f[X^q]_{\mathrm{supercell}} + E_{\mathrm{correction}}.

In the limit of an infinitely large supercell, the formation energy in the
supercell should approach that of the infinite cell, and the correction should
go to zero. To accomplish this in two dimensions, we must be careful about how
we scale the cell - If we increase the lateral size of the supercell without
changing the vacuum, or if we increase the vacuum without changing the size
of the supercell, we get incorrect results. Because of these difficulties,
finding a robust correction scheme in 2D is that much more necessary. The
starting point is the same as before, and we write:

.. math::
  E[X^q] - E_0  = (E[X^q] - E_0)_{\mathrm{uncorrected}} - E_{\mathrm{l}} + q\Delta V.

The main problem arises in the `E_{\mathrm{l}}` term, and in particular, in
finding an appropriate expression for the screened Coulomb interaction of the
system.

Recall that `E_{\mathrm{l}}` consists of two terms: The energy (per unit
cell) associated with the interaction between the model charge distribution
and all its periodic images, and the energy of the isolated charge
distribution in the same dielectric environment. The first step in
calculating either of these is to model the dielectric function.

The approach used here is to assume that the dielectric function of the
isolated layer is isotropic in plane, and varies only in the `z` direction.
Additionally, the screening is assumed to follow the density distribution of
the system so that we can write

.. math::
 \varepsilon_{i}(z) = k_i\cdot n(z) + 1,

Where `i` varies over "in-plane" and "out-of-plane", and `n` is the in-plane
averaged density of the system. The normalization constants, `k_i`, are
chosen such that

.. math::

  \frac{1}{L} \int \mathrm{d} z\, \varepsilon_{\parallel}(z) &= \varepsilon_{\parallel}^{\mathrm{DFT}} \\
  \frac{1}{L} \int \mathrm{d} z\, \varepsilon_{\perp}^{-1}(z) &= \left(\varepsilon_{\perp}^{\mathrm{DFT}}\right)^{-1}

The quantities appearing on the right hand side are the in-plane and
out-of-plane dielectric constants resulting from a DFT/RPA calculation on the
smallest unit cel of the pristine system, but with the same amount of vacuum as
the supercell calculation. For 15 Å of vacuum, we calculate
`\varepsilon_{\parallel}^{\mathrm{DFT}} = 1.80` and
`\varepsilon_{\perp}^{\mathrm{DFT}} = 1.14`. The resulting dielectric profiles are shown below

.. figure:: dielectric_profile.png

With the dielectric functions in hand, we can proceed to solve the poisson
equation, and calculate the two interaction energies. For more detail on how
exactly the potentials and energies of the periodic and isolated charge
distributions are calculated, we refer to :ref:`defects_theory`.

The following script gives the example of a carbon boron substitution in
hexagonal boron nitride:

.. literalinclude:: BN.py

This script sets up a carbon boron substitution in an NxN supercell of the
4-atom rectangular cell, and calculates the energy of the +1 charge state, the
neutral defect and the pristine system, saving the intermediate gpw files.

For the 2x2 system with a vacuum of 15 Å we calculate the uncorrected total
energy difference `(E[X^q] - E_0)_\mathrm{uncorrected}` to be 4.1 eV.

The electrostatic corrections have been implemented in the following script,
which greatly resembles the equivalent script for the GaAs defect. The two
differences are:

- The ``dimensionality=2d`` keyword argument to electrostatic correction
  constructor, which ensures that the object uses the 2D model for the
  dielectric functions.
- The use of a list in the ``set_epsilons()`` member function, rather than a
  single number. The first item corresponds to the in-plane dielectric
  constant, while the second corresponds to the out-of-plane dielectric
  constant.

.. literalinclude:: electrostatics_BN.py

As before, we can calculate the formation energy of the charged defect as a
function of the supercell size, both with and without the electrostatic
corrections. Since all of these systems have the same amount of vacuum, it is
not expected that the uncorrected values converge to the true result, but it is
encouraging to see the behaviour of the corrected values. As we are dealing
with an unrelaxed system, we expect the formation energy of the defect to be
too high, but again, the focus is on the electrostatics and the qualitative
behaviour as a function of the supercell size, rather than the true values.

.. figure:: energies_BN.png
            :scale: 50 %

As a test of the quality of the electrostatic corrections, we can look at the
charge transition level between the neutral state and the +1 state. This is the
energy with respect to the valence band at which the two formation energies are
equal, and can be calculated just as `E_f[X^0] - E_f[X^{+1}]`. It comes to 2
eV, in good agreement with [#KomsaErratum]_.


References
==========

.. [#RMP]  C. Freysoldt et al.
              *Rev. Mod. Phys.* **86**, 253 (2014)

.. [#Lany] S. Lany and A. Zunger
              *Phys. Rev. B* **78**, 235104 (2008)

.. [#FNV] C. Freysoldt, J. Neugebauer and C. G. Van de Walle
              *Phys. Status Solidi B* **248**, 1067 (2011)

.. [#Komsa] H.-P. Komsa, T. T. Rantala and A. Pasquarello
              *Phys. Rev. B* **86**, 045112 (2012)

.. [#KomsaErratum] H.-P. Komsa, N. Berseneva, A. V. Krasheninnikov and
              R. M. Nieminen
              *Phys Rev X* **4**, 031044 (2014)
