.. _ps2ae:

Obtaining all-electron wave functions and electrostatic potential
=================================================================

Wave functions
--------------

To get from the pseudo (PS) wave function to the all-electron (AE) wave
function, a PAW correction needs to be added.  The correction will add the
cusp and all the wiggles necessary for the wave function to be orthogonal to
all the frozen core states.  This tutorial show how to use the
:class:`~gpaw.utilities.ps2ae.PS2AE` class for interpolating the PS wave
function to a fine grid where the PAW corrections can be represented:

.. autoclass:: gpaw.utilities.ps2ae.PS2AE

Here is the code for plotting some AE wave functions for a HLi dimer using a
PAW dataset for Li with a frozen 1s orbital
(:meth:`~gpaw.utilities.ps2ae.PS2AE.get_wave_function`):

.. literalinclude:: hli_wfs.py

.. figure:: hli-wfs.png

.. automethod:: gpaw.utilities.ps2ae.PS2AE.get_wave_function


Electrostatic potential
-----------------------

The relevant formulas can be found here: :ref:`electrostatic potential`.

Here is how to extract the AE potential from a gpw-file using the
:meth:`~gpaw.utilities.ps2ae.PS2AE.get_electrostatic_potential` method:

.. literalinclude:: hli_pot.py

.. figure:: hli-pot.png

.. automethod:: gpaw.utilities.ps2ae.PS2AE.get_electrostatic_potential
