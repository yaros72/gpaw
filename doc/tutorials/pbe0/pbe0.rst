.. _pbe0_tut:

==================================
PBE0 calculations for bulk silicon
==================================

This tutorial will do non-selfconsistent PBE0 based on self-consistent PBE.

.. seealso::

   * :ref:`bandstructures` tutorial.
   * :ref:`band exercise` exercice.


PBE and PBE0 band gaps
======================

The band structure can be calculated like this:

.. literalinclude:: gaps.py

These are the resulting `\Gamma`-`\Gamma` and `\Gamma`-`X` gaps for PBE and PBE0 in eV:
    
.. csv-table::
    :file: si-gaps.csv
    :header: **k**-points, PBE(G-G), PBE(G-X), PBE0(G-G), PBE0(G-X)
             

Lattice constant and bulk modulus
=================================

Here is how to calculate the lattice constant:
    
.. literalinclude:: eos.py

Plot the results like this:
    
.. literalinclude:: plot_a.py

.. image:: si-a.png
