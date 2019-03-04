.. _curie:

===========================================================
Curie   (BullX cluster, Intel Nehalem, Infiniband QDR, MKL)
===========================================================

.. note::
      These instructions are up-to-date as of October 2014

Here you find information about the the system
http://www-hpc.cea.fr/en/complexe/tgcc-curie.htm

For large calculations, it is suggested that one utilizes the Scalable Python
interpreter. Small to medium size calculations are fine with standard Python,
for these one can use system's default Python (which contains NumPy),
thus one can skip directly to LibXC/GPAW instructions.


Scalable Python
===============

Standard Python interpreter has serious bottleneck in large scale parallel
calculations when many MPI tasks perform the same I/O operations during the
import statetements. Scalable Python `<https://gitorious.org/scalable-python>`_
reduces the import time by having only single MPI task to perform import
related I/O operations and using then MPI for broadcasting the data.

First, download the source code and switch to GNU environment::

  git clone git@gitorious.org:scalable-python/scalable-python.git
  module switch intel gnu
  export OMPI_MPICC=gcc

Use the following build script (change the installation prefix to a
proper one):

.. literalinclude:: build_scalable_python_curie.sh

Add then ``install_prefix/bin`` to your PATH, and download and install NumPy::

   export PATH=install_prefix/bin:$PATH
   wget http://sourceforge.net/projects/numpy/files/NumPy/1.8.1/numpy-1.8.1.tar.gz
   tar xzf numpy-1.8.1.tar.gz
   cd numpy-1.8.1
   python setup.py install

LibXC
=====

Download libxc::

   wget http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-2.2.0.tar.gz

Configure and make::

   ./configure --prefix=install_prefix CFLAGS=-fPIC
   make
   make install

GPAW
====

Intel compiler gives a bit better performance for GPAW, so one should switch
back to Intel environment after Scalable Python/LibXC installation::

    module switch gnu intel
    unset OMPI_MPICC

Furthermore, in order to utilize HDF5 load the module::

    module load hdf5/1.8.12_parallel

Use the compiler wrapper file :git:`~doc/platforms/Linux/icc.py`

.. literalinclude:: icc.py

and the following configuration file
:download:`customize_curie_icc.py`.

.. literalinclude:: customize_curie_icc.py

GPAW can now build in a normal way (make sure Scalable Python is in the PATH)::

    python setup.py install --home=path_to_install_prefix

