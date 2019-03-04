.. _neb:

=========================================
NEB calculations parallelized over images
=========================================

The :ref:`ase:diffusion tutorial` example can be used with GPAW like
this:

.. literalinclude:: neb.py

If we run the job on 12 cpu's::

  $ mpirun -np 12 gpaw-python neb.py

then each of the three internal images will be parallelized over 4 cpu's.

The results are read with::

  $ ase gui neb.traj@-5:

The energy barrier is found to be ~0.3 eV for LDA.
