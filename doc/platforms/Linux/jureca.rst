.. _jureca:

==================
jureca @ FZ-Jülich
==================

Find information about the `Jureca system here`_.

.. _Jureca system here: http://www.fz-juelich.de/ias/jsc/jureca

Pre-installed versions
======================

You may use the pre-installed versions::

  module load intel-para
  module load GPAW

In case you are happy with these versions, you need to install
the setups (next point) and you are done.

Setups
======

The setups are not defined in the pre-installed vesrion, so we need
to install them ourselves::

  cd
  GPAW_SETUP_SOURCE=$PWD/source/gpaw-setups
  mkdir -p $GPAW_SETUP_SOURCE
  cd $GPAW_SETUP_SOURCE
  wget https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-0.9.11271.tar.gz
  tar xzf gpaw-setups-0.9.11271.tar.gz
  
Let gpaw know about the setups::
  
  export GPAW_SETUP_PATH=$GPAW_SETUP_SOURCE/gpaw-setups-0.9.11271

Using the module environment
============================

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

Building from trunk
===================

In case that you need a newer version than is installed, you might want 
to install gpaw yourself.

We first get ASE trunk::

  ASE_SOURCE=$HOME/source/ase
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

  if {![is-loaded intel-para]} {module load intel-para}
  if {![is-loaded Python]} {module load Python}
  if {![is-loaded SciPy-Stack]} {module load SciPy-Stack}

  # customize the following according to your taste
  set HOME $env(HOME)

  set asehome $HOME/source/ase/trunk
  prepend-path       PYTHONPATH    $asehome
  prepend-path       PATH          $asehome/bin:$asehome/tools

We create a place for gpaw and get the trunk version::

  # customize the following according to your taste
  GPAW_SOURCE=$HOME/source/gpaw

  mkdir -p $GPAW_SOURCE
  cd $GPAW_SOURCE
  git clone https://gitlab.com/gpaw/gpaw.git trunk

The current trunk version can then be updated by::

  cd $GPAW_SOURCE/trunk
  git pull

We use the installed version of libxc and our ase/trunk::

  module purge
  module load intel-para
  module load Libxc
  module load ase/trunk

and install using
:download:`customize_jureca.py`::

  cd $GPAW_SOURCE/trunk
  mkdir install
  cp customize_jureca.py customize.py
  python setup.py install --prefix=$PWD/install

We add this also to the module environment::

  cd
  mkdir -p modulefiles/gpaw
  cd modulefiles/gpaw
  
and the module file  :file:`trunk` should read::

  #%Module1.0

  if {![is-loaded ase/trunk]} {module load ase}
  if {![is-loaded Libxc]} {module load Libxc}
  if {![is-loaded gpaw-setups]}  {module load gpaw-setups}

  # customize the following according to your taste
  set HOME $env(HOME)

  set gpawhome $HOME/source/gpaw/trunk/install
  prepend-path    PATH                 $gpawhome/bin
  prepend-path    PYTHONPATH           $gpawhome/lib/python3.6/site-packages
  setenv          GPAW_PYTHON          $gpawhome/bin/gpaw-python


Execution
=========

Job scripts can be written using::

  gpaw-runscript -h

and the jobs sumitted as::
    
  sbatch run.jureca
