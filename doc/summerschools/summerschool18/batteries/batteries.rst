.. _batteries:

=========
Batteries
=========

In this exercise we will study the anode and cathode material of a Li-ion
battery. The cathode material will be |LiFePO4| a typical cathode material in
rechargeable Li-ion batteries. The anode will be graphite.

The first day we start out soft by calculating the intercalation energy of Li
in graphite while we learn the methods and workflow using ASE and GPAW. The
second day will be about determining the equilibrium potential of a
|LiFePO4|/C battery, we will also use a Bayesian approach to estimate the DFT
error we expect on this important value. On the final day we will determine
important battery characteristics such Li transport barriers and the voltage
profile.

Tools used:

* Structure creation and modification with ASE

* Unit cell relaxation

* Bayesian error estimation

* Nudged Elastic Band (NEB) calculations for estimating Li migration barriers


Part 1: Li intercalation energy in graphite
===========================================

:download:`batteries1.ipynb`, :download:`C64.png`,
:download:`Li2.png`, :download:`C144Li18.png`

The notebook ``batteries1.ipynb`` will guide you through the first day of the
battery exercise.

* Setup a graphite structure

* Calculate C-C and interlayer distances

  - Use an empirical potential and DFT with a couple of exchange correlation
    functionals and compare with experimental values

* Setup and calculate the energy of Li metal

  - Using DFT only from now on

* Setup and calculate the combined structure of Li between graphene layers

* Use all values to determine the Li intercalation energy

  - Compare the results of different functionals with experimental values.


Part 2: Equilibrium potential of a |LiFePO4|/C battery
======================================================

:download:`batteries2.ipynb`, :download:`lifepo4_wo_li.traj`

You will calculate the equilibrium potential and use Bayesian error estimation
to quantify how sensitive the calculated equilibrium potential is towards
choice of functional. The notebook is ``batteries2.ipynb``.

* Setup and calculate |FePO4| and |LiFePO4| structures

  - Use these and the previous Li metal calculation to determine the
    equilibrium potential of a |FePO4|/Li battery

* Get an uncertainty estimation on the potential by using an ensemble of
  functionals called a ``BEEFEnsemble``

* Using values from the previous day calculate the equilibrium potential of
  the full Li |FePO4|/C battery


Part 3: Transport barriers and voltage profile
==============================================

:download:`batteries3.ipynb`, :download:`NEB_init.traj`

You will calculate the energy barriers for transport of Li intercalated in the
graphite anode. You will examine how sensitive this barrier is to the
interlayer distance in graphite. You will also examine the energy of
intermediate states during the charge/discharge process. This will allow some
basic discussion of the voltage profile of the battery. The notebook is
``batteries3.ipynb``.

* Create intial and final structures for a NEB calculation, that will
  determine the transition state

  - If time permits you can study the influence of changing the interlayer
    graphite distance on the energy barrier.

* Create structures for a Li vacancy in |LiFePO4| and a single Li in |FePO4|

* Calculate the Li vacancy/insertion energies and compare them to the
  equilibrium potential

  - What can they tell you about the charge/discharge potential curves?


 .. |FePO4| replace:: FePO\ :sub:`4`

 .. |LiFePO4| replace:: LiFePO\ :sub:`4`
