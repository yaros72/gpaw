.. _magnetism:

===============
Magnetism in 2D
===============

This exercise investigates magnetic order in 2D. While, a magnetic ground
state may be found for a 2D material using DFT, magnetic order at finite
temperatures requires spin-orbit coupling and magnetic anisotropy.

The exercise will teach you how to extract magnetic exchange and anisotropy
parameters from first principles calculations. It will also touch upon the
Mermin-Wagner theorem and show why anisotropy is crucial for magnetic order in
2D. The first part shows how to calculate the Curie temperature in |CrI3|. In
the second part you you investigate |VI2|, which has anti-ferromagnetic
coupling and non-collinear order. In the third part you will search for a new
2D material with large critical temperature based on a database of 2D
materials.


Part 1: Critical temperaure of |CrI3|
=====================================

:download:`magnetism1.ipynb`, :download:`CrI3.xyz`

The notebook ``magnetism1.ipynb`` shows how to set up a monolayer of |CrI3| and
calculate the critical temperature

* Set up a the structure and optimize the geometry of |CrI3|

* Calculate the exchange parameter from a total energy mapping analysis

* Derive the instability of the magnetic ground state when anisotropy is
  neglected (The Mermin-Wagner theorem)

* Calculate the magnetic anisotropy and critical temperature


Part 2: Non-collinear magnetism - |VI2|
=======================================

:download:`magnetism2.ipynb`, :download:`VI2.xyz`

If the magnetic atoms form a hexagonal lattice and the exchange coupling is
anti-ferromagnetic, the ground state will have a non-collinear structure. In
the notebook ``magnetism2.ipynb`` you will

* Relax the atomic postions of the material

* Compare a collinear anti-ferromagnetic structure with the ferromagnetic state

* Obtain the non-collinear ground state

* Calculate the magnetic anisotropy and discuss whether or not the mateerial
  will exhibit magnetic order at low temperature


Part 3: Find a new 2D material with large critical temperature
==============================================================

:download:`magnetism3.ipynb`

In this last part you will search the database and pick one material you
expect to have a large critical temperature. The total energy mapping analysis
is carried out to obtain exchange coupling parameters and a first principles
estimate of the critical temperature. The guidelines for the analysis is found
in the notebook ``magnetism3.ipynb``.

.. |CrI3| replace:: CrI\ :sub:`3`

.. |VI2| replace:: VI\ :sub:`2`
