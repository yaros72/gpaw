.. _optimizer_tests:

===============
Optimizer tests
===============

This page shows benchmarks of optimizations done with ASE's different
:mod:`optimizers <ase.optimize>`.
Note that the iteration number (steps) is not the same as the number of force
evaluations. This is because some of the optimizers uses internal line searches
or similar.

The most important performance characteristics of an optimizer is the
total optimization time.
Different optimizers may perform the same number of steps, but along a different
path, so the time spent on calculation of energy/forces may be different
due to different convergence of the self-consistent field.


Test systems
============

These are the test systems (:download:`systems.db`):

.. csv-table::
   :file: systems.csv
   :header-rows: 1


EMT calculations
================

Calculation done with :class:`~ase.calculators.emt.EMT`.  Number of steps:

.. csv-table::
   :file: emt-iterations.csv
   :header-rows: 1


GPAW-LCAO calculations
======================

Parameters::

    GPAW(mode='lcao',
         basis='dzp',
         kpts={'density': 2.0})

Absolute time relative to fastest optimizer:

.. csv-table::
   :file: lcao-time.csv
   :header-rows: 1
