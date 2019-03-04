====================================
Manual installation on Ubuntu 18.04+
====================================

Install these Ubuntu_ packages::

    $ sudo apt install python3-dev libopenblas-dev liblapack-dev libxc-dev libscalapack-mpi-dev libblacs-mpi-dev

Then install ASE_, Numpy and SciPy::

    $ python3 -m pip install ase

And finally, GPAW with ScaLAPACK::

    $ wget https://pypi.org/packages/source/g/gpaw/gpaw-1.4.0.tar.gz
    $ tar -xf gpaw-1.4.0.tar.gz
    $ cd gpaw
    $ sed -i "s/scalapack = False/scalapack = True/" customize.py
    $ python3 setup.py install --user


.. _Ubuntu: http://www.ubuntu.com/
.. _ASE: https://wiki.fysik.dtu.dk/ase/
