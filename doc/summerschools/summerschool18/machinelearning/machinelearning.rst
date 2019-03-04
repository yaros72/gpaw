.. _machinelearning:

================
Machine Learning
================

In this exercise, you will experience a few machine learning methods for design of
new materials starting from an existing database. In particular, you will design
a model to identify good candidate materials for light harvesting, based on a
small database of organic/inorganic perovskites. Afterwards, you will validate
the model predictions by running DFT calculations on selected systems.

Part 1: Inspection of database
==============================

:download:`machinelearning.ipynb`, :download:`organometal.db`

The first part of the exercise is an inspection of the existing database.
Understanding what is available from other sources is a necessary step before
running any machine learning tools. Here, you will:

* Extract structures from the database.

* Calculate heat of formations.

* Plot histograms and scatter plots for different quantities available from
  the database.


Part 2: Machine Learning
========================

In this part, you will implement the machine learning model:

* Define the input vectors.

* Select a suitable functional form with optimal parameters.

* Find a loss function to evaluate the performances.

* Apply this model to the prediction of the heat of formation.

* Improve the input vectors and the model.

Part 3: Test and Evaluate the Model
===================================

In the last part, you will test the prediction model and run DFT calculations
for the heat of formation and the band gap to compare these results with the
model.

