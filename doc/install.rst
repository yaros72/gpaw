.. _installation:

============
Installation
============

.. toctree::
    :hidden:

    troubleshooting
    platforms/platforms

GPAW relies on the Python library *atomic simulation environment* (ASE_),
so you need to :ref:`install ASE <ase:download_and_install>` first.  GPAW
itself is written mostly in the Python programming language, but there
are also some C-code used for:

* performance critical parts
* allowing Python to talk to external numerical libraries (BLAS_, LAPACK_,
  LibXC_, MPI_ and ScaLAPACK_)

So, in order to make GPAW work, you need to compile some C-code.  For serial
calculations, you will need to build a dynamically linked library
(``_gpaw.so``) that the standard Python interpreter can load.  For parallel
calculations, you need to build a new Python interpreter (``gpaw-python``)
that has MPI_ functionality built in.

There are several ways to install GPAW:

* On a lucky day it's as simple as ``pip3 install -U gpaw`` as
  described :ref:`below <installation using pip>`.

* Alternatively, you can :ref:`download <download>` the source code,
  edit :git:`customize.py` to tell the install script which libraries you
  want to link to and where they
  can be found (see :ref:`customizing installation`) and then install with a
  ``python3 setup.py install --user`` as described :ref:`here <install
  with distutils>`.

* There may be a package for your Linux distribution that you can use
  (named ``gpaw``).

* If you are a developer that need to change the code you should look at this
  description: :ref:`developer installation`.

.. seealso::

    * Using :ref:`homebrew` on MacOSX.
    * Using :ref:`anaconda`.
    * Tips and tricks for installation on many :ref:`platforms and
      architectures`.
    * :ref:`troubleshooting`.
    * Important :ref:`envvars`.
    * In case of trouble: :ref:`Our mail list and IRC channel <contact>`.


Requirements
============

* Python_ 2.7, 3.4-
* NumPy_ 1.9 or later (base N-dimensional array package)
* SciPy_ 0.14 or later (library for scientific computing)
* ASE_ 3.17.0 or later (atomic simulation environment)
* a C-compiler
* LibXC_ 3.x or 4.x
* BLAS_ and LAPACK_ libraries

Optional, but highly recommended:

* an MPI_ library (required for parallel calculations)
* FFTW_ (for increased performance)
* BLACS_ and ScaLAPACK_


.. _Python: http://www.python.org/
.. _NumPy: http://docs.scipy.org/doc/numpy/reference/
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _LibXC: http://www.tddft.org/programs/libxc/
.. _MPI: http://www.mpi-forum.org/
.. _BLAS: http://www.netlib.org/blas/
.. _BLACS: http://www.netlib.org/blacs/
.. _LAPACK: http://www.netlib.org/lapack/
.. _ScaLAPACK: http://www.netlib.org/scalapack/
.. _PyPI: https://pypi.org/project/gpaw
.. _PIP: https://pip.pypa.io/en/stable/
.. _ASE: https://wiki.fysik.dtu.dk/ase
.. _FFTW: http://www.fftw.org/


.. _installation using pip:

Installation using ``pip``
==========================

.. highlight:: bash

The simplest way to install GPAW is using pip_ and the GPAW package from
the Python package index (PyPI_)::

    $ pip3 install --upgrade --user gpaw

This will compile and install GPAW (both ``_gpaw.so`` and all the Python
files) in your ``~/.local/lib/pythonX.Y/site-packages`` folder where
Python can automatically find it.  The ``pip3`` command will also place
the command line tool :command:`gpaw` in the ``~/.local/bin`` folder, so
make sure you have that in your :envvar:`PATH` environment variable.  If
you have an ``mpicc`` command on your system then there will also be a
``gpaw-python`` executable in ``~/.local/bin``.

Check that you have installed everything in the correct places::

    $ gpaw info

To check the compiled parallel features (like ScaLAPACK), you need to run::

    $ gpaw-python -m gpaw info


Install PAW datasets
====================

Install the datasets into the folder ``<dir>`` using this command::

    $ gpaw install-data <dir>

See :ref:`installation of paw datasets` for more details.

Now you should be ready to use GPAW, but before you start, please run the
tests as described below.


.. index:: test
.. _run the tests:

Run the tests
=============

Make sure that everything works by running the test suite::

    $ gpaw test

This will take a couple of hours.  You can speed it up by using more than
one core::

    $ gpaw test -j 4

Please report errors to the ``gpaw-users`` mailing list so that we
can fix them (see :ref:`mail list`).

If tests pass, and the parallel version is built, test the parallel code::

    $ gpaw -P 4 test

or equivalently::

    $ mpiexec -np 4 gpaw-python -m gpaw test


.. _download:

Getting the source code
=======================

Sou can get the source from a tar-file or from Git:

:Tar-file:

    You can get the source as a tar-file for the
    latest stable release (gpaw-1.5.1.tar.gz_) or the latest
    development snapshot (`<snapshot.tar.gz>`_).

    Unpack and make a soft link::

        $ tar -xf gpaw-1.5.1.tar.gz
        $ ln -s gpaw-1.5.1 gpaw

    Here is a `list of tarballs <https://pypi.org/simple/gpaw/>`__.

:Git clone:

    Alternatively, you can get the source for the latest stable release from
    https://gitlab.com/gpaw/gpaw like this::

        $ git clone -b 1.5.1 https://gitlab.com/gpaw/gpaw.git

    or if you want the development version::

        $ git clone https://gitlab.com/gpaw/gpaw.git

Add ``~/gpaw`` to your :envvar:`PYTHONPATH` environment variable and add
``~/gpaw/tools`` to :envvar:`PATH` (assuming ``~/gpaw`` is where your GPAW
folder is).

.. note::

    We also have Git tags for older stable versions of GPAW.
    See the :ref:`releasenotes` for which tags are available.  Also the
    dates of older releases can be found there.

.. _gpaw-1.5.1.tar.gz:
    https://pypi.org/packages/source/g/gpaw/gpaw-1.5.1.tar.gz


.. _customizing installation:

Customizing installation
========================

The install script does its best when trying to guess proper libraries
and commands to build GPAW. However, if the standard procedure fails
or user wants to override default values it is possible to customize
the setup with :git:`customize.py` file which is located in the GPAW base
directory. As an example, :git:`customize.py` might contain the following
lines::

  libraries = ['myblas', 'mylapack']
  library_dirs = ['path_to_myblas']

Now, GPAW would be built with "``-Lpath_to_myblas -lmyblas
-lmylapack``" linker flags. Look at the file :git:`customize.py`
itself for more possible options.  :ref:`platforms and architectures`
provides examples of :file:`customize.py` for different platforms.
After editing :git:`customize.py`, follow the instructions for the
:ref:`developer installation`.


.. _install with distutils:

Install with setup.py
=====================

If you have the source code, you can use the install script (:git:`setup.py`)
to compile and install the code::

    $ python3 setup.py install --user


.. _parallel installation:

Parallel installation
=====================

By default, setup looks if :program:`mpicc` is available, and if setup
finds one, a parallel version is build. If the setup does not find
mpicc, a user can specify one in the :git:`customize.py` file.

Additionally a user may want to enable ScaLAPACK, setting in
:git:`customize.py`::

    scalapack = True

and, in this case, provide BLACS/ScaLAPACK ``libraries`` and ``library_dirs``
as described in :ref:`customizing installation`.

Instructions for running parallel calculations can be found in the
:ref:`user manual <manual_parallel_calculations>`.


FFTW
====

The FFTW library is linked at runtime using :mod:`ctypes`.  By default, the
following shared library names are searched for: ``libmkl_rt.so``,
``libmkl_intel_lp64.so`` and ``libfftw3.so``.  First one found will be
loaded.  If no library is found, the :mod:`numpy.fft` library will be used
instead.  The name of the FFTW shared library can also be set via the
``$GPAW_FFTWSO`` environment variable.


.. _libxc installation:

Libxc Installation
==================

If you OS does not have a LibXC package you can use then you can download
and install LibXC as described `here
<http://www.tddft.org/programs/libxc/>`_.  A
few extra tips:

* Libxc installation requires both a C compiler and a fortran compiler.

* We've tried intel and gnu compilers and haven't noticed much of a
  performance difference.  Use whatever is easiest.

* Libxc shared libraries can be built with the "--enable-shared" option
  to configure.  This might be slightly preferred because it reduces
  memory footprints for executables.

* Typically when building GPAW one has to modify customize.py in a manner
  similar to the following::

    library_dirs += ['/my/path/to/libxc/4.2.3/install/lib']
    include_dirs += ['/my/path/to/libxc/4.2.3/install/include']

  or if you don't want to modify your customize.py, you can add these lines to
  your .bashrc::

    export C_INCLUDE_PATH=/my/path/to/libxc/4.2.3/install/include
    export LIBRARY_PATH=/my/path/to/libxc/4.2.3/install/lib
    export LD_LIBRARY_PATH=/my/path/to/libxc/4.2.3/install/lib

Example::

    wget http://www.tddft.org/programs/octopus/down.php?file=libxc/4.2.3/libxc-4.2.3.tar.gz -O libxc-4.2.3.tar.gz
    tar -xf libxc-4.2.3.tar.gz
    cd libxc-4.2.3
    ./configure --enable-shared --disable-fortran --prefix=$HOME/libxc-4.2.3
    make
    make install

    # add these to your .bashrc:
    XC=~/libxc-4.2.3
    export C_INCLUDE_PATH=$XC/include
    export LIBRARY_PATH=$XC/lib
    export LD_LIBRARY_PATH=$XC/lib


.. _envvars:

Environment variables
=====================

.. envvar:: PATH

    Colon-separated paths where programs can be found.

.. envvar:: PYTHONPATH

    Colon-separated paths where Python modules can be found.

.. envvar:: OMP_NUM_THREADS

    Currently should be set to 1.

.. envvar:: GPAW_SETUP_PATH

    Comma-separated paths to folders containing the PAW datasets.

Set these permanently in your :file:`~/.bashrc` file::

    $ export PYTHONPATH=~/gpaw:$PYTHONPATH
    $ export PATH=~/gpaw/tools:$PATH

or your :file:`~/.cshrc` file::

    $ setenv PYTHONPATH ${HOME}/gpaw:${PYTHONPATH}
    $ setenv PATH ${HOME}/gpaw/tools:${PATH}
