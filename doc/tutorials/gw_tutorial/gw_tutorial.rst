.. module:: gpaw.response.g0w0
.. _gw tutorial:

=========================================================
Quasi-particle spectrum in the GW approximation: tutorial
=========================================================

For a brief introduction to the GW theory and the details of its
implementation in GPAW, see :ref:`gw_theory`.

More information can be found here:

    \F. Hüser, T. Olsen, and K. S. Thygesen

    `Quasiparticle GW calculations for solids, molecules, and
    two-dimensional materials`__

    Physical Review B, Vol. **87**, 235132 (2013)

    __ http://prb.aps.org/abstract/PRB/v87/i23/e235132


Quasi-particle spectrum of bulk diamond
=======================================

In the first part of the tutorial, the G0W0 calculator is introduced and the
quasi-particle spectrum of bulk diamond is calculated.


Groundstate calculation
-----------------------

First, we need to do a regular groundstate calculation. We do this in plane
wave mode and choose the LDA exchange-correlation functional. In order to
keep the computational efforts small, we start with (3x3x3) k-points and a
plane wave basis up to 300 eV.

.. literalinclude:: C_groundstate.py

It takes a few seconds on a single CPU. The last line in the script creates a
.gpw file which contains all the informations of the system, including the
wavefunctions.

.. note::

    You can change the number of bands to be written out by using
    ``calc.diagonalize_full_hamiltonian(nbands=...)``.
    This can be useful if not all bands are needed.


The GW calculator
-----------------

Next, we set up the G0W0 calculator and calculate the quasi-particle spectrum
for all the k-points present in the irreducible Brillouin zone from the ground
state calculation and the specified bands.
In this case, each carbon atom has 4 valence electrons and the bands are double
occupied. Setting ``bands=(3,5)`` means including band index 3 and 4 which is
the highest occupied band and the lowest unoccupied band.

.. literalinclude:: C_gw.py

It takes about 30 seconds on a single CPU for the
:meth:`~gpaw.response.g0w0.G0W0.calculate` method to finish:

.. automethod:: gpaw.response.g0w0.G0W0.calculate

The dictionary is stored in ``C-g0w0_results.pckl``.  From the dict it is
for example possible to extract the direct bandgap at the Gamma point:

.. literalinclude:: get_gw_bandgap.py

with the result: 6.96 eV.

The possible input parameters of the G0W0 calculator are listed here:

.. autoclass:: gpaw.response.g0w0.G0W0


Convergence with respect to cutoff energy and number of k-points
-----------------------------------------------------------------

Can we trust the calculated value of the direct bandgap? Not yet. A check for
convergence with respect to the plane wave cutoff energy and number of k
points is necessary. This is done by changing the respective values in the
groundstate calculation and restarting. Script
:download:`C_ecut_k_conv_GW.py` carries out the calculations and
:download:`C_ecut_k_conv_plot_GW.py` plots the resulting data. It takes
several hours on a single xeon-8 CPU (8 cores). The resulting figure is
shown below.

.. image:: C_GW.png
    :height: 400 px

A k-point sampling of (8x8x8) seems to give results converged to within 0.05 eV.
The plane wave cutoff is usually converged by employing a `1/E^{3/2}_{\text{cut}}` extrapolation.
This can be done with the following script: :download:`C_ecut_extrap.py` resulting
in a direct band gap of 7.57 eV. The extrapolation is shown in the figure below

.. image:: C_GW_k8_extrap.png
    :height: 400 px


Frequency dependence
--------------------

Next, we should check the quality of the frequency grid used in the
calculation. Two parameters determine how the frequency grid looks.
``domega0`` and ``omega2``. Read more about these parameters in the tutorial
for the dielectric function :ref:`df_tutorial_freq`.

Running script :download:`C_frequency_conv.py` calculates the direct band
gap using different frequency grids with ``domega0`` varying from 0.005 to
0.05 and ``omega2`` from 1 to 25. The resulting data is plotted in
:download:`C_frequency_conv_plot.py` and the figure is shown below.

.. image:: C_freq.png
    :height: 400 px

Converged results are obtained for ``domega0=0.02`` and ``omega2=15``, which
is close to the default values.


Final results
-------------

A full G0W0 calculation with (8x8x8) k-points and extrapolated to infinite cutoff results in a direct band gap of 7.57 eV. Hence the value of 6.96 eV calculated at first was not converged!

Another method for carrying out the frequency integration is the Plasmon Pole
approximation (PPA). Read more about it here :ref:`gw_theory_ppa`. This is
turned on by setting ``ppa = True`` in the G0W0 calculator (see
:download:`C_converged_ppa.py`). Carrying out a full `G_0W_0` calculation with the PPA
using (8x8x8) k-points and extrapolating from calculations at a cutoff of 300
and 400 eV gives a direct band gap of 7.52 eV, which is in very good agreement
with the result for the full frequency integration but the calculation took
only minutes.

.. note::

    If a calculation is very memory heavy, it is possible to set ``nblocks``
    to an integer larger than 1 but less than or equal to the amount of CPU
    cores running the calculation. With this, the response function is divided
    into blocks and each core gets to store a smaller matrix.

.. _gw-2D:

Quasi-particle spectrum of two-dimensional materials
====================================================
Carrying out a G0W0 calculation of a 2D system follows very much the same recipe
as outlined above for diamond. To avoid having to use a large amount of vacuum in
the out-of-plane direction we advice to use a 2D truncated Coulomb interaction,
which is turned on by setting ``truncation = '2D'``. Additionally it is possible
to add an analytical correction to the q=0 term of the Brillouin zone sampling
by specifying ``q0_correction=True``. This means that a less dense k-point
grid will be necessary to achieve convergence. More information about this
specific method can be found here:

    \F. A. Rasmussen, P. S. Schmidt, K. T. Winther and K. S. Thygesen

    `Efficient many-body calculations for two-dimensional materials using exact limits for the screened potential: Band gaps of MoS2, h-BN and phosphorene`__

    Physical Review B, Vol. **94**, 155406 (2016)

    __ https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.155406

How to set up a 2D slab of MoS2 and calculate the band structure can be found in
:download:`MoS2_gs_GW.py`. The results are not converged but a band gap of 2.57 eV is obtained.

Including vertex corrections
============================
Vertex corrections can be included through the use of a xc kernel known from TDDFT. The vertex corrections can be included in the polarizability and/or the self-energy. It is only physically well justified to include it in both quantities simultaneously. This leads to the `GW\Gamma` method. In the `GW\Gamma` method, the xc kernel mainly improves the description of short-range correlation which manifests itself in improved absolute band positions. Only including the vertex in the polarizability or the self-energy results in the `GWP` and `GW\Sigma`  method respectively. All three options are available in GPAW. The short-hand notation for the self-energy in the four approximations available is summarized below:

.. math:: &\text{GW:}\quad \Sigma^{GW} = iGv(1-\chi_0v)^{-1}\\
 &\text{GWP:}\quad \Sigma^{GWP} = iGv(1-\chi_0f_{xc})(1-\chi_0(v+f_{xc}))^{-1}\\
 &\text{GW}\Gamma\text{:}\quad \Sigma^{GW\Gamma} = iGv(1-\chi_0(v+f_{xc}))^{-1}\\
 &\text{GW}\Sigma\text{:}\quad \Sigma^{GW\Sigma} = iGv(1 + \chi_0(1-v\chi_0)^{-1}(v+f_{xc}))

More information can be found here:

    \P. S. Schmidt, C. E. Patrick, and K. S. Thygesen

    `Simple vertex correction improves GW band energies of bulk and
    two-dimensional crystals`__

    To appear in Physical Review B.

    __ https://arxiv.org/abs/1711.02922

.. note::
    Including vertex corrections is currently not possible for spin-polarized systems.

A `GW\Gamma` calculation requires that 3 additional keywords are specified in the GW calculator: 

1) Which kernel to use: ``xc='rALDA'``, ``xc='rAPBE'`` etc.. 

2) How to construct the kernel: ``av_scheme='wavevector'`` or ``av_scheme='density'``. The wavevector scheme is preferred here. 

3) How to apply the kernel: ``fxc_mode = 'GWG'``, ``fxc_mode='GWP'`` or ``fxc_mode='GWS'``.

Carrying on from the ground state calculation in :download:`MoS2_gs_GW.py`, a `GW\Gamma` calculation can be done with the following script: :download:`MoS2_GWG.py`.

The `GW` and `GW\Gamma` band structures can be visualized with the :download:`MoS2_bs_plot.py` script resulting in the figure below. Here, the effect of the vertex is to shift the bands upwards by around 0.5 eV whilst leaving the band gap almost unaffected.

.. image:: MoS2_bs.png
    :height: 400 px

.. note::
    When carrying out a `G_0W_0\Gamma` calculation by specifying the 3 keywords above, the ``do_GW_too = True`` option allows for a simultaneous `G_0W_0` calculation. This is faster than doing two seperate calculations as `\chi_0` only needs to be calculated once, but the memory requirement is twice that of a single `G_0W_0` calculation. The `G_0W_0\Gamma` results will by default be stored in g0w0_results.pckl and the `G_0W_0` results in g0w0_results_GW.pckl. The results of both calculations will be printed in the output .txt file.

.. _gw-GW0:

GW0 calculations
================
It is currently possible to add eigenvalue self-consistency in the Green's function.
This is activated by setting ``method='GW0'``, specifying how many iterations you
are interested in, ``maxiter=5`` and optionally also how much of the previous iteration's
eigenvalues you want mixed in, ``mixing=0.5`` . Usually 5 iterations are enough to reach
convergence with a mixing of 50%. Only the bands specified will be updated self-consistently.
The bands above(below) the highest(lowest) included band will be shifted with the
k-point averaged shift of the band below(above).
The results after each iteration is printed in the output file.
The following script calculates the band gap of bulk BN within the GW0 approximation:
:download:`BN_GW0.py`. The figure below shows the value of the gap during the first five
iterations, where iteration zero is the DFT gap (:download:`BN_GW0_plot.py`).
Note: The calculations are not converged with respect to k-points, frequency points or cutoff energy.

.. image:: BN_GW0.png
    :height: 400 px
