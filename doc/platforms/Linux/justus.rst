======
justus
======

Information about `justus <https://www.bwhpc-c5.de/wiki/index.php/BwForCluster_Chemistry>`__.

Building GPAW
=============

We assume that the installation will be located in ``$HOME/source``.

Setups
------

The setups must be installed first::

  cd
  GPAW_SETUP_SOURCE=$PWD/source/gpaw-setups
  mkdir -p $GPAW_SETUP_SOURCE
  cd $GPAW_SETUP_SOURCE
  wget https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.11271.tar.gz
  tar xzf gpaw-setups-0.9.11271.tar.gz

Let gpaw know about the setups::

  export GPAW_SETUP_PATH=$GPAW_SETUP_SOURCE/gpaw-setups-0.9.11271

Using the module environment
----------------------------

It is very handy to add our installation to the module environment::

  cd
  mkdir -p modulefiles/gpaw-setups
  cd modulefiles/gpaw-setups
  echo -e "#%Module1.0\nprepend-path       GPAW_SETUP_PATH    $GPAW_SETUP_SOURCE/gpaw-setups-0.9.11271" > 0.9.11271

We need to let the system know about our modules
(add this command to ``~/.profile`` or ``~/.bashrc`` to execute automatically)::

  module use $HOME/modulefiles

such that we also see them with::

  module avail

libxc
-----

GPAW relies on libxc (see the `libxc web site <http://www.tddft.org/programs/octopus/wiki/index.php/Libxc:download>`__).
To install libxc we assume that ``MYLIBXCDIR`` is set to
the directory where you want to install
(e.g. ``MYLIBXCDIR=$HOME/source/libxc``)::

 mkdir -p $MYLIBXCDIR
 cd $MYLIBXCDIR
 wget http://www.tddft.org/programs/octopus/down.php?file=libxc/libxc-2.2.0.tar.gz -O libxc-2.2.0.tar.gz
 tar xzf libxc-2.2.0.tar.gz
 cd libxc-2.2.0
 mkdir install
 ./configure CFLAGS="-fPIC" --prefix=$PWD/install -enable-shared
 make |tee make.log
 make install

This will have installed the libs ``$MYLIBXCDIR/libxc-2.0.2/install/lib``
and the C header
files to ``$MYLIBXCDIR/libxc-2.0.2/install/include``.
We create a module for libxc::

  cd
  mkdir modulefiles/libxc
  cd modulefiles/libxc

and edit the module file  :file:`2.2.0` that should read::

  #%Module1.0

  #                                    change this to your path
  set             libxchome            /home/fr/fr_fr/fr_mw767/source/libxc/libxc-2.2.0/install
  prepend-path    C_INCLUDE_PATH       $libxchome/include
  prepend-path    LIBRARY_PATH         $libxchome/lib
  prepend-path    LD_LIBRARY_PATH      $libxchome/lib

ASE release
-----------

You might want to install a stable version of ASE::

  cd
  ASE_SOURCE=$PWD/source/ase
  mkdir -p $ASE_SOURCE
  cd $ASE_SOURCE
  git clone -b 3.13.0 https://gitlab.com/ase/ase.git 3.13.0

We add our installation to the module environment::

  cd
  mkdir -p modulefiles/ase
  cd modulefiles/ase

Edit the module file  :file:`3.13.0` that should read::

  #%Module1.0

  if {![is-loaded numlib/python_scipy]} {module load numlib/python_scipy}

  #           change this to your path
  set asehome /home/fr/fr_fr/fr_mw767/source/ase/3.13.0
  prepend-path       PYTHONPATH    $asehome
  prepend-path       PATH          $asehome/tools

ASE trunk
---------

We get ASE trunk::

  cd
  ASE_SOURCE=$PWD/source/ase
  mkdir -p $ASE_SOURCE
  cd $ASE_SOURCE
  git clone https://gitlab.com/ase/ase.git trunk

which can be updated using::

  cd $ASE_SOURCE/trunk
  git pull

We add our installation to the module environment::

  cd
  mkdir -p modulefiles/ase
  cd modulefiles/ase

and edit the module file  :file:`trunk` that should read::

  #%Module1.0

  if {![is-loaded numlib/python_scipy]} {module load numlib/python_scipy}

  #           change this to your path
  set asehome /home/fr/fr_fr/fr_mw767/source/ase/trunk
  prepend-path       PYTHONPATH    $asehome
  prepend-path       PATH          $asehome/tools

Building GPAW
-------------

We create a place for gpaw and get the trunk version::

 cd
 GPAW_SOURCE=$PWD/source/gpaw
 mkdir -p $GPAW_SOURCE
 cd $GPAW_SOURCE
 git clone https://gitlab.com/gpaw/gpaw.git trunk

The current trunk version can then be updated by::

 cd $GPAW_SOURCE/trunk
 git pull

A specific tag can be loaded by::

 cd $GPAW_SOURCE/trunk
 # list tags
 git tag
 # load version 1.2.0
 git checkout 1.2.0

We have to modify the file :file:`customize.py` to
:download:`customize_justus.py`

.. literalinclude:: customize_justus.py

To build GPAW use::

 module purge
 module load libxc
 module load ase
 module load mpi/impi

 cd $GPAW_SOURCE/trunk
 mkdir install
 python3 setup.py install --prefix=$PWD/install

which installs GPAW to ``$GPAW_SOURCE/trunk/install``.
We create a module that does the necessary things::

  cd
  mkdir -p modulefiles/gpaw
  cd modulefiles/gpaw

the file  :file:`trunk` that should read::

 #%Module1.0

 if {![is-loaded ase]} {module load ase}
 if {![is-loaded libxc]} {module load libxc}
 if {![is-loaded mpi]}  {module load mpi/impi}
 if {![is-loaded gpaw-setups]}  {module load gpaw-setups}

 # change this to your needs
 set gpawhome /home/fr/fr_fr/fr_mw767/source/gpaw/trunk/install
 prepend-path    PATH                 $gpawhome/bin
 prepend-path    PYTHONPATH           $gpawhome/lib/python3.5/site-packages/
 setenv          GPAW_PYTHON          $gpawhome/bin/gpaw-python

Running GPAW
------------

A gpaw script :file:`test.py` can be submitted to run on 8 cpus like this::

  > module load gpaw
  > gpaw-runscript test.py 8
  using justus
  run.justus written
  > msub run.justus
