.. _ensembles:

=========
Ensembles
=========

In the tutorial on :ref:`atomization_energy`, we calculated the
atomization energy for H\ `2`:sub: using the PBE functional.  In this
tutorial, we wish to estimate the uncertainty of the calculated
atomization energy.

In the following script, an ensemble of 1000 atomization
energies are calculated (non-selfconsistently) with an ensemble of 1000
exchange-correlation functionals distributed according to their
probability (see article [Mor05b]_ for details).

.. literalinclude:: ensembles.py

The script produces a text file named ensemble_energies.txt.

* You must run the :ref:`atomization <atomization_energy>` script first.

Ensemble of atomization energies for H\ `2`:sub: :

.. image:: ensemble.png
   :width: 400 px


.. [Mor05b] J. J. Mortensen, K. Kaasbjerg, S. L. Frederiksen,
   J. K. Nørskov, J. P. Sethna, and K. W. Jacobsen,
   Phys. Rev. Lett. 95, 216401 (2005)
