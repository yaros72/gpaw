=========
Muon Site
=========

Positive muons implanted in metals tend to stop at interstitial sites that
correspond to the maxima of the Coulomb potential energy for electrons in the
material. In turns the Coulomb potential is approximated by the Hartree
pseudo-potential obtained from the GPAW calculation. A good guess is therefore
given by the maxima of this potential.

In this tutorial we obtain the guess in the case of MnSi. The results can be
compared with A. Amato et al. [Amato14]_, who find a muon site at fractional
cell coordinates (0.532,0.532,0.532) by DFT calculations and by the analysis
of experiments.


MnSi calculation
================

Let's perform the calculation in ASE, starting from the space group of MnSi,
198, and the known Mn and Si coordinates.

.. literalinclude:: mnsi.py
   :end-before: assert

The ASE code outputs a Gaussian cube file, mnsi.cube, with volumetric data of
the potential (in eV) that can be visualized.


Getting the maximum
===================

One way of identifying the maximum is by the use of an isosurface (or 3d
contour surface) at a slightly lower value than the maximum. This can be done
by means of an external visualization program, like eg. majavi
(see also :ref:`iso surface`)::

    $ python3 -m ase.visualize.mlab -c 11.1,13.3 mnsi.cube

The parameters after -c are the potential values for two countour surfaces
(the maximum is 13.4 eV).

This allows also secondary (local) minima to be identified.

A simplified procedure to identify the global maximum is the following

.. literalinclude:: plot2d.py

The figure below shows the contour plot of the pseudo-potential in the plane
z=2.28 Angstrom containing the maximum

.. image:: pot_contour.png

The absolute maximum is at the center of the plot, at (2.28,2.28,2.28), in
Angstrom. A local maximum is also visible around (0.6,1.75,2.28), in Angstrom.

In comparing with [Amato14]_ keep in mind that the present examples has a very
reduced number of k points and a low plane wave cutoff energy, just enough to
show the right extrema in the shortest CPU time.


-------------

.. [Amato14]  A. Amato et al.
   Phys. Rev. B. 89, 184425 (2014)
   *Understanding  the μSR spectra of MnSi without magnetic polarons*
   DOI: 10.1103/PhysRevB.1.4555
