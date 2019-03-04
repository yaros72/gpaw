.. _homebrew:

========
Homebrew
========

.. highlight:: bash

Get Xcode from the App Store and install it. You also need to install the
command line tools, do this with the command::

    $ xcode-select --install

Install Homebrew::

    $ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    $ echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bash_profile

Install ASE and GPAW dependencies::

    $ brew install python
    $ brew install gcc
    $ brew install libxc
    $ brew install open-mpi
    $ brew install fftw
    $ brew install pygtk

Install pip::

    $ sudo easy_install pip

Install required Python packages::

    $ pip install numpy scipy matplotlib

Install and test ASE::

    $ pip install --upgrade --user ase
    $ python -m ase test

Install GPAW::

    $ pip install --upgrade --user gpaw

Install GPAW setups::

    $ gpaw --verbose install-data

.. note::

  Alternative solution if the above command fails::

    $ curl -O https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.20000.tar.gz
    $ tar -xf gpaw-setups-0.9.20000.tar
    $ echo 'export GPAW_SETUP_PATH=~/gpaw-setups-0.9.20000' >> ~/.bash_profile

Test GPAW::

    $ gpaw test
    $ gpaw -P 4 test
