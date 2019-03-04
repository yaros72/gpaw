.. _ehrenfest:

=============================
Ehrenfest dynamics (TDDFT/MD)
=============================

For a brief introduction to the Ehrenfest dynamics theory and the details of
its implementation in GPAW, see :ref:`Ehrenfest theory <ehrenfest_theory>`.
The original implementation by Ari Ojanpera is described in
Ref. [#Ojanpera2012]_.


.. seealso::

    * :ref:`timepropagation`
    * :class:`gpaw.tddft.TDDFT`
    * :meth:`gpaw.tddft.TDDFT.propagate`


------------
Ground state
------------

Similar to static TDDFT calculations, one has to start with a standard ground
state simulation. In TDDFT, one can use larger grid spacing than for geometry
optimization, so for example, if you use ``h=0.25`` for geometry optimization,
try ``h=0.3`` for TDDFT to save a lot of time (the same spacing should be
used for the ground state and TDDFT calculators).

Ground state example:

.. literalinclude:: h2_gs.py

Ehrenfest TDDFT/MD is also available in :ref:`LCAO mode <lcaotddft>`.


--------------------------
Simulating H2 dissociation
--------------------------

We then simulate Ehrenfest dynamics of the H2 molecule in a very intense
laser field to observe its dissociation.

For Ehrenfest dynamics we must use the parameter ``propagator='EFSICN'`` for
the TDDFT calculator ``tdcalc`` to take into account the necessary
corrections in the propagation of the time-dependent Kohn-Sham equation. The
parameter ``solver='BiCGStab'`` to use the stabilized biconjugate gradient
method (``BiCGStab``) is generally recommended for Ehrenfest dynamics
calculations for any system more complex than simple dimers or dimers.

H2 dissociation example:

.. literalinclude:: h2_diss.py

The distance between the H atoms at the end of the dynamics is more than 2 Å 
and thus their bond was broken by the intense laser field.


-------------------------------
Electronic stopping in graphene
-------------------------------

A more complex use for Ehrenfest dynamics is to simulate the irradiation of 
materials with either chared ions (Ref. [#Ojanpera2014]_) or neutral atoms 
(Ref. [#Brand2019]_).

The following script calculates the ground state of the projectile + target
system, with the parameter ``charge`` defining its charge state. For ionisation
state +1, an external potential is used at the hydrogen ion the converge the
calculation. One might also have to change the default convergence parameters
depending on the projectile used, and to verify the convergence of the results 
with respect to the timestep and *k*-points. Here, slightly less strict criteria 
are used. The impact point in this case is the center of a carbon hexagon, but 
this can be modified by changing the x-y position of the H atom
(``projpos``).

Projectile + target example:

.. literalinclude:: graphene_h_gs.py

Finally, the following script can be used for performing an electronic
stopping calculation for a hydrogen atom impacting graphene with the initial
velocity being 40 keV. In the charged state, the external potential is 
automatically set to zero when the TDDFT object is initialized and hence does 
not affect the calculation. The calculation ends when the distance between the 
projectile and the bottom of the supercell is less than 5 Å. (Note: this is a 
fairly demanding calculation with 8 cores and requires close to 50 GB of 
memory.)

Electronic stopping example:

.. literalinclude:: graphene_h_prop.py


----------
References
----------

.. [#Ojanpera2012] A. Ojanpera, V. Havu, L. Lehtovaara, M. Puska,
                   "Nonadiabatic Ehrenfest molecular dynamics within
                   the projector augmented-wave method",
                   *J. Chem. Phys.* **136**, 144103 (2012).

.. [#Ojanpera2014] A. Ojanpera, Arkady V. Krasheninnikov, M. Puska,
                   "Electronic stopping power from first-principles
                   calculations with account for core
                   electron excitations and projectile ionization",
                   *Phys. Rev. B* **89**, 035120 (2014).

.. [#Brand2019] C. Brand et al., to be published.
