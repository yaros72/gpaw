.. _load on niflheim:

======================================
Using a pre-installed GPAW at Niflheim
======================================

This is the guide for using the pre-installed GPAW modules on Niflheim.

Modules on Niflheim
===================

You can see which modules are available with the ``module avail [package]`` command, for example::

  $ module avail GPAW

  --------------------------- /home/modules/modules/all ---------------------------
     GPAW/1.3.0-foss-2017b-Python-3.6.3         GPAW-setups/0.8.7929
     GPAW/1.4.0-foss-2018a-Python-3.6.4  (D)    GPAW-setups/0.9.9672
     GPAW/1.4.0-foss-2018b-Python-3.6.6         GPAW-setups/0.9.11271
     GPAW/1.4.0-intel-2018b-Python-3.6.6        GPAW-setups/0.9.20000 (D)

    Where:
     D:  Default Module

You can see which modules you have loaded with ``module list``.  You
can unload all modules to start from scratch with ``module purge``.


Choose the right version of GPAW
================================

This is a brief guide to which version of GPAW you should use. It
reflects the situation in December 2018 and will soon be updated as
the situation changes.

I am very conservative
  You should load ``GPAW/1.4.0-foss-2018a-Python-3.6.4``

  This will give you exactly the same build of GPAW that you have been
  using during most of 2018. This will, however, break ``gedit``,
  ``emacs``, ``gnuplot`` and possibly some other graphical programs.
  This is because it loads ``fontconfig/2.12.6-GCCcore-6.4.0`` which
  is incompatible with the version of CentOS installed on the login
  nodes.

I am moderately conservative
  You should load ``GPAW/1.4.0-foss-2018b-Python-3.6.6``

  This will give you the same *version* of GPAW that you have been
  using all year, but a *new build*, compiled with a newer compiler
  and a newer Python.  It will not crash your editor.

I want the fastest version
  You should load ``GPAW/1.4.0-intel-2018b-Python-3.6.6`` and be prepared
  to update to GPAW version 1.5.0 when released.

  This will give the same version of GPAW, but compiled with the Intel
  compilers.  This should in general give better performance, in
  particular on the ``xeon24`` nodes.  For a standard DFT calculation
  expect 10-20% speedup on ``xeon16`` and 20-30% on ``xeon24``.  Some
  operations may be slower, for example diagonalization to find many
  empty bands on the ``xeon16``.

**IMPORTANT:**  You do *not* need to load Python, ASE, matplotlib etc.
Loading GPAW pulls all that stuff in, in versions consistent with the
chosen GPAW version.


Module consistency is important: check it.
==========================================

For a reliable computational experience, you need to make sure that
all modules come from the same toolchain (i.e. that the software is
compiled with a consistent set of tools).  **All modules you
load should belong to the same toolchain.**

Use ``module list`` to list your modules. Check for consistency:

* If you use the ``foss-2018a`` toolchain, all modules should end in
  ``foss-2018a``, ``foss-2018a-Python-3.6.4``, ``gompi-2018a`` or
  ``GCCcore-6.4.0``.

* If you use the ``foss-2018b`` toolchain, all modules should end in
  ``foss-2018b``, ``foss-2018b-Python-3.6.6``, ``gompi-2018b`` or
  ``GCCcore-7.3.0``.

* If you use the ``intel-2018b`` toolchain, all modules should end in
  ``intel-2018b``, ``intel-2018b-Python-3.6.6``, ``gompi-2018b`` or
  ``GCCcore-7.3.0``.

If your ``module load XXX`` commands give warnings about reloaded
modules, you are almost certainly mixing incompatible toolchains.


Using different versions for different projects.
================================================

You do not have to use the same modules for all your projects.  If you
want all jobs submitted from the folder ``~/ProjectAlpha`` to run with
an one version of GPAW, but everything else with a another version,
you can put this in your .bashrc::

  if [[ $SLURM_SUBMIT_DIR/ = $HOME/tmp/gpaw-foss* ]]; then
      # Extreme consistency is important for this old project
      module purge
      module load GPAW/1.4.0-foss-2018a-Python-3.6.4
  else
      # Performance is important for everything else.
      module load GPAW/1.4.0-intel-2018b-Python-3.6.6
      module load scikit-learn/0.20.0-intel-2018b-Python-3.6.6.eb
  fi

The ``module purge`` command in the special branch is because SLURM
will remember which modules you have loaded when you submit the job,
and that will typically be the default version, which must then be
unloaded.
