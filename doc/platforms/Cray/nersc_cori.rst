.. _nersc_cori:

=====================
cori.nersc.gov (XC40)
=====================

.. note::
   These instructions are up-to-date as of July 2018.

GPAW
====

At NERSC it is recommened to install GPAW on Cori with Anaconda python. For
massivly parallel applications it is recommened to use `Shifter
<http://www.nersc.gov/research-and-development/user-defined-images/>`_.

GPAW can be built with a minimal ``customize.py``

.. literalinclude:: customize_nersc_cori.py

Load the GNU programming environment and set Cray environment for dynamic
linking::

  export CRAYPE_LINK_TYPE=dynamic
  module swap ${PE_ENV,,} PrgEnv-gnu
  module load python

Create a conda environment for gpaw::

  conda create --name gpaw python=3.6 pip numpy scipy matplotlib

Install ASE with pip while the Anaconda python module is loaded::

  source activate gpaw
  pip install ase

Build and install GPAW::

  python setup.py build_ext
  python setup.py install

To setup the environment::

  module swap ${PE_ENV,,} PrgEnv-gnu
  module load python
  source activate gpaw
  export OMP_NUM_THREADS=1

Then the test suite can be run from a batch script or interactive session with::

  export MKL_CBWR="AVX"
  srun -n 8 -c 2 --cpu_bind=cores gpaw-python -m gpaw test

.. note::
   For all tests to pass enable MKL's conditional Numerical
   Reproducibility mode with the `MKL_CBWR` environment variable.
