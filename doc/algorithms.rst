.. _features and algorithms:

=======================
Features and algorithms
=======================

Quick links to all features:

.. list-table::

    * - :ref:`Plane-waves <manual_mode>`
      - :ref:`Finite-difference <manual_stencils>`
      - :ref:`LCAO <lcao>`
    * - :ref:`XC-functionals <xc>`
      - :ref:`DFT+U <hubbardu>`
      - :ref:`GLLB-SC <bandgab>`
    * - :ref:`DOS <pdos>`
      - :ref:`STM <stm tutorial>`
      - :ref:`Wannier functions <wannier>`
    * - :ref:`delta-SCF <dscf>`
      - :ref:`XAS <xas>`
      - :ref:`Jellium <jellium>`
    * - :ref:`TDDFT <timepropagation>`
      - :ref:`LRTDDFT (molecules) <lrtddft>`
      - :ref:`LRTDDFT (extended systems) <df_theory>`
    * - :ref:`RPA-correlation <rpa>`
      - :ref:`GW <gw_theory>`
      - :ref:`BSE <bse theory>`
    * - :ref:`Parallelization <parallel_runs>`
      - :ref:`Continuum Solvent Model <continuum_solvent_model>`
      -

This Page gives a quick overview of the algorithms used.  We have
written some :ref:`papers <gpaw_publications>` about the implementation,
where *all* the details can be found.


Introduction
============

Using the projector-augmented wave (PAW)
method [Blo94]_, [Blo03]_  allows us to get rid of the core
electrons and work with soft pseudo valence wave functions.  The
pseudo wave functions don't need to be normalized - this is important
for the efficiency of calculations involving 2. row elements (such as
oxygen) and transition metals.  A further advantage of the PAW method
is that it is an all-electron method (frozen core approximation) and
there is a one to one transformation between the pseudo and
all-electron quantities.


Description of the wave functions
=================================

Pseudo wave functions can be described in three ways:

Finite-difference (FD):
    Uniform real-space orthorhombic grids.  Two kinds of grids are involved
    in the calculations: A coarse grid used for the wave functions and a fine
    grid (`2^3=8` times higher grid point density) used for densities and
    potentials.  The pseudo electron density is first calculated on the coarse
    grid from the wave functions, and then interpolated to the fine grid, where
    compensation charges are added for achieving normalization.  The effective
    potential is evaluated on the fine grid (solve the Poisson equation and
    calculate the exchange-correlation potential) and then restricted to the
    coarse grid where it needs to act on the wave functions (also on the coarse
    grid).

Plane-waves (PW):
    Expansion in plane-waves.  There is one cutoff used for the wave-functions
    and a higher cutoff for electron densities and potentials.

Linear combination of atomic orbitals (LCAO):
    Expansion in atom-centered basis functions.


Grid-based techniques for FD-mode
=================================

Solving the Kohn-Sham equation is done via iterative multi-grid
eigensolvers starting from a good guess for the wave functions
obtained by diagonalizing a Hamiltonian for a subspace of atomic orbitals.
We use the multi-grid preconditioner described by Briggs *et al.* [Bri96]_
for the residuals, and standard Pulay mixing is used to update the density.


Compensation charges
====================

Compensation charges
are expanded to give correct multipole moments up to angular momentum
number `\ell=2`.


Boundary conditions
===================

In each of the three directions, the boundary conditions can be either
periodic or open.


Mask function technique
=======================

Due to the discreticed nature of space in finite difference methods,
the energy of an atom will depend on its position relative to the grid
points.  The problem comes from the calculation of the integral of a
wave function times an atom centered localized function (radial
functions times a spherical harmonic).  To reduce this dependence, we
use the technique of [Taf06]_, where the radial functions (projector functions) are smoothened as follows:

* Divide function by a mask function that goes smoothly to zero at
  approximately twice the cutoff radius.
* Fourier transform.
* Cut off short wavelength components.
* Inverse Fourier transform.
* Multiply by mask function.


Exchange-correlation functionals
================================

All the functionals from the :ref:`libxc <xc_functionals>` library can
be used.  Calculating the XC-energy and potential for the extended
pseudo density is simple.  For GGA functionals, a nearest neighbor
finite difference stencil is used for the gradient operator.  In the
PAW method, there is a correction to the XC-energy inside the
augmentation spheres.  The integration is done on a non-linear radial
grid - very dense close to the nuclei and less dense away from the
nuclei.


Parallelization
===============

Parallelization is done by distributing **k**-points, spins, and bands
over all processors and on top of that domain-decomposition is used.


ASE interface
=============

The code has been designed to work together with the atomic
simulation environment (`ASE <https://wiki.fysik.dtu.dk/ase>`). ASE provides:

 * Structure optimization.
 * Molecular dynamics.
 * Nudged elastic band calculations.
 * Maximally localized Wannier functions.
 * Scanning tunneling microscopy images.


Open Software
=============

GPAW is released under the `GNU Public License <http://xkcd.com/225>`_
version 3 or any later version.  See the file :git:`COPYING` which
accompanies the downloaded files, or see the license at GNU's web
server at http://www.gnu.org/licenses/.  Everybody is invited to
participate in using and :ref:`developing the code <devel>`.


.. figure:: carlsberg.png
    :width: 12cm

    September 2003 - August 2005: Sponsored by The `Carlsberg Foundation`_
    (artwork by P. Erhart)

.. _Carlsberg Foundation: http://www.carlsbergfondet.dk


.. [Blo94] P. E. Blöchl,
   Phys. Rev. B 50, 17953 (1994)
.. [Blo03] P. E. Blöchl, C. J. Först and J. Schimpl,
   Bull. Mater. Sci, 26, 33 (2003)
.. [Bri96] E. L. Briggs, D. J. Sullivan and J. Bernholc,
   Phys. Rev. B 54, 14362 (1996)
.. [Taf06] *A general and efficient pseudopotential Fourier filtering scheme
   for real space methods using mask functions*, Maxim Tafipolsky, Rochus
   Schmid, J Chem Phys. 2006 May 7;124:174102
