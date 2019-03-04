.. _build on niflheim:

=========================
Building GPAW on Niflheim
=========================

Information about the Niflheim cluster can be found at
`<https://wiki.fysik.dtu.dk/niflheim>`_.

This document explains how to compile a developer version of GPAW on
Niflheim.  If you just want to run the pre-installed version, please
read the guide :ref:`Using a pre-installed GPAW on Niflheim <load on niflheim>`.


.. highlight:: bash

Get the ASE and GPAW source code
================================

Here, we install the development versions of ASE and GPAW in ``~/ase`` and
``~/gpaw``.  Make sure the folders do not exist, remove them if they
do (or update with ``git pull`` if you have done this step previously)::

    $ cd
    $ git clone https://gitlab.com/ase/ase.git
    $ git clone https://gitlab.com/gpaw/gpaw.git

	       
Choose the right compiler suite
===============================

GPAW has traditionally been compiled with the standard GNU compilers,
but compiling with the Intel compiler suite and the Intel Math Kernel
Library (MKL) gives approximately 30% better performance.

**We strongly recommend using the Intel compiler!**  To do so, put
this in your .bashrc::

  ASE=~/ase
  GPAW=~/gpaw
  source $GPAW/doc/platforms/Linux/Niflheim/gpaw-intel.sh

The first two lines specify where you keep your developer checkouts of
ASE and GPAW.

If you insist on using the GNU compilers (at a **performance
penalty**) do this instead::

  ASE=~/ase
  GPAW=~/gpaw
  source $GPAW/doc/platforms/Linux/Niflheim/gpaw-foss.sh

Note that these files import Python and matplotlib compiled with the
same compilers as you will use for GPAW.  This means that any other
module you import should be with the same compiler (intel or foss).
For example if you need scikit-learn, it should be imported as
``scikit-learn/0.20.0-intel-2018b-Python-3.6.6`` to get the intel
version.  The files above sets a variable ``$MYPYTHON`` to make this
easier, and you can import e.g. scikit-learn with::

  module load scikit-learn/0.20.0-$MYPYTHON   # Just an example


Optional: Select libxc and perhaps libvdwxc
===========================================

Per default, GPAW is compiled with libxv version 3.0.1 since there
appear to be rare circumstances where newer libxc versions give wrong
results (probably only related to users generating their own PAW
setups).

If you want a newer version you should load it explicitly *before*
sourcing the ``gpaw-intel.sh`` or ``gpaw-foss.sh`` file::

  ASE=~/ase
  GPAW=~/gpaw
  module load libxc/4.2.3-intel-2018b
  source $GPAW/doc/platforms/Linux/Niflheim/gpaw-intel.sh

Similarly, if you need XC potentials from ``libvdwxc`` you should manually load
this module as well.  It is currently only available with the foss
compiler suite due to an incompatibility between the FFTW and MKL
libraries.  **IMPORTANT:** GPAW support van der Waals correlation
even without ``libvdwxc`` loaded!




Installing GPAW on all Niflheim architectures
=============================================

Compile GPAW's C-extension using the :download:`compile.sh` script::

    $ source ~/.bashrc    # Only needed if you changed your .bashrc file.
    $ cd gpaw
    $ sh doc/platforms/Linux/Niflheim/compile.sh

Submit jobs to the queue with::

    $ gpaw sbatch -- -p xeon8 -N 2 -n 16 my-script.py

Type ``gpaw sbatch -h`` for help.


Using more than one version of GPAW
===================================

Here, we install an additional version of GPAW for, say, test runs::

    $ cd ~
    $ mkdir testing
    $ cd testing
    $ ... clone gpaw and compile ...

Add this to your ``~/.bashrc``::

    if [[ $SLURM_SUBMIT_DIR/ = $HOME/test-runs* ]]; then
        GPAW=~/testing/gpaw
    fi

right before sourcing the ``gpaw-foss.sh`` or ``gpaw-intel.sh`` script
mentioned above. Now, SLURM-jobs submitted inside your ``~/test-runs/``
folder will use the version of GPAW from the ``~/testing/`` folder.

Using more than one compiler with GPAW
======================================

If we want to run different versions of GPAW with different compilers, we
again make an additional clone of the GPAW repository::

    $ cd ~
    $ mkdir performancetest
    $ cd performancetest
    $ ... clone gpaw ...

Say you normally use the foss compiler and want to try out the intel one
for performance. Then we have to create the right bash environment, both
before compiling and upon ssh'ing into each login node. The latter is
done by specifying what bash commands the compile script should run
immediately after ssh'ing into each node. These commands can be given to
the compile script as inputs::

    $ module purge
    $ GPAW=~/performancetest/gpaw
    $ source $GPAW/doc/platforms/Linux/Niflheim/gpaw-intel.sh
    $ cd ~/performancetest/gpaw
    $ sh doc/platforms/Linux/Niflheim/compile.sh 'module purge' 'GPAW=~/performancetest/gpaw' 'source $GPAW/doc/platforms/Linux/Niflheim/gpaw-intel.sh'

Instead of typing all your commands into the terminal, you can write them
in a file. Say you write a file ``~/perfomancetest/gpaw-intel-env.sh``::

  module purge
  GPAW=~/performancetest/gpaw
  source $GPAW/doc/platforms/Linux/Niflheim/gpaw-intel.sh
  module list

where the ``module list`` command has been added to track your modules.
Then you can compile your performance test version of gpaw by::

    $ source ~/performancetest/gpaw-intel-env.sh
    $ cd ~/performancetest/gpaw
    $ sh doc/platforms/Linux/Niflheim/compile.sh 'source ~/performancetest/gpaw-intel-env.sh'

Finally, you need to load the right modules, when you are using the
performance test version of gpaw. This is done in your ``~/.bashrc``
by choosing the specific compiler toolchain together with the
version of gpaw::

  if [[ $SLURM_SUBMIT_DIR/ = $HOME/performancetest-runs* ]]; then
      GPAW=~/performancetest/gpaw
      source $GPAW/doc/platforms/Linux/Niflheim/gpaw-intel.sh
  fi

  if [[ -z $GPAW ]]; then
      GPAW=~/gpaw
      source $GPAW/doc/platforms/Linux/Niflheim/gpaw-foss.sh
  fi
