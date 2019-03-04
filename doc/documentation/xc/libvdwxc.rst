.. _libvdwxc-doc:

libvdwxc
========

`libvdwxc <https://gitlab.com/libvdwxc/libvdwxc>`_
is a library which provides fast and scalable
implementations of non-local van der Waals density functionals in the
vdW-DF family.
To use libvdwxc, you need to install it
and compile GPAW with it.  libvdwxc can be used with other semilocal
functionals like optPBE, optB88, and BEEF-vdW.

`Install <http://libvdwxc.readthedocs.io>`_ libvdwxc,
making sure that its dependencies FFTW3 and
FFTW3-MPI are available on the system.  For truly large systems, you
may install PFFT to achieve better scalability, but FFTW3-MPI may well
be more efficient except for very large systems.

Currently there is no stable libvdwxc release yet.  Clone the project from
git and install manually.

Run a calculation as follows:

.. literalinclude:: libvdwxc-example.py

libvdwxc will automatically parallelize with as many cores as are
available for domain decomposition.  If you parallelize over *k*-points
or bands, and *especially* if you use planewave mode, be sure to pass
the parallelization keyword ``augment_grids=True`` to make use of *all*
cores including those for *k*-point and band parallelization.

Here is a more complex example:

.. literalinclude:: libvdwxc-pfft-example.py

Normally you should probably not bother to set pfft_grid as it is
chosen automatically.
