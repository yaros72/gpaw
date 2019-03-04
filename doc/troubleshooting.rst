.. _troubleshooting:

Troubleshooting
===============

XXX todo!

::

    gpaw info

If tests pass, and the parallel version is built, test the parallel code::

    $ mpiexec -np 2 gpaw-python -c "import gpaw.mpi as mpi; print(mpi.rank)"
    1
    0

.. note::

   Many MPI versions have their own ``-c`` option which may
   invalidate python command line options. In this case
   test the parallel code as in the example below.

Try also::

    $ ase build H -V 2 | gpaw -P 2 run -p mode=pw

This will perform a calculation for a single spin-polarized hydrogen atom
parallelized with spin up on one processor and spin down on the other.
