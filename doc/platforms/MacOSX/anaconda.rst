.. _anaconda:

=================
Anaconda on MacOS
=================

We recommend using Python from :ref:`homebrew` on macOS, but if you need to use the Anaconda python that is also possible.  Both ASE and GPAW work with Anaconda python.

Install Anaconda
================

* We recommend installing the Python 3 version (but Python 2 should work, we have not tested it, however).

* We strongly recommend installing Anaconda for a single user.  The permission handling in Anaconda is broken on macOS, and a multi-user installation of Anaconda will break as soon as another user installs a package.

Install Homebrew
================

You need it for some GPAW prerequisites!

Get the Xcode Command Line Tools with the command::

    $ xcode-select --install

(if it fails, you may have to download Xcode from the App Store)
    
Install Homebrew::

    $ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    $ echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bash_profile

Install ASE and GPAW dependencies
=================================

::
   
    $ brew install libxc
    $ brew install open-mpi
    $ brew install fftw

Check your installation
=======================

You should still get python and pip from the anaconda installation::

    $ python --version
    $ pip --version

Python should list an anaconda version, and pip should be loaded from ``/Users/xxxx/anaconda3/....``

Install and test ASE and GPAW
=============================

Install and test ASE::
   
    $ pip install --upgrade --user ase
    $ python -m ase test

Install GPAW::

    $ pip install --upgrade --user gpaw

Install GPAW setups::

    $ gpaw --verbose install-data

    
