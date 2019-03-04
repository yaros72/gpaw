.. _dipole:

================================
Dipole-layer corrections in GPAW
================================

As an example system, a 2 layer `2\times2` slab of fcc (100) Al
is constructed with a single Na adsorbed on one side of the surface.

.. literalinclude:: dipole.py
    :lines: 1-14

.. image:: slab.png

The :func:`ase.build.fcc100` function will create a slab
with periodic boundary conditions in the xy-plane only and GPAW will
therefore use zeros boundary conditions for the the wave functions and
the electrostatic potential in the z-direction as shown here:

.. image:: zero.png

The blue line is the xy-averaged potential and the green line is the
fermi-level.  See below how to extract the potential using the
:meth:`~gpaw.paw.PAW.get_electrostatic_potential` method.

If we use periodic boundary conditions in all directions:

.. literalinclude:: dipole.py
    :lines: 15-18

the electrostatic potential will be periodic and average to zero:

.. image:: periodic.png

In order to estimate the work functions on the two sides of the slab,
we need to have flat potentials or zero electric field in the vacuum
region away from the slab.  This can be achieved by using a dipole
correction:

.. literalinclude:: dipole.py
    :lines: 20-24

.. image:: corrected.png

In PW-mode, the potential must be periodic and in that case the corrected
potential looks like this (see [Bengtsson]_ for more details):

.. image:: pwcorrected.png

See the full Python scripts here :download:`dipole.py` and here
:download:`pwdipole.py`.  The script
used to create the figures in this tutorial is shown here:

.. literalinclude:: plot.py

.. [Bengtsson] Lennart Bengtsson,
    Dipole correction for surface supercell calculations,
    Phys. Rev. B 59, 12301 - Published 15 May 1999
