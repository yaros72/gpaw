.. _bader analysis:

==============
Bader Analysis
==============

Henkelman *et. al* have implemented a fast and robust algorithm for
calculating the electronic charges on individual atoms in molecules or
crystals, based on the Bader partitioning scheme [Bader]_. In that
method, the analysis is based purely on the electron density. The
partitioning of the density is determined according to its zero-flux
surfaces. Details of their implementation can be found in [Tang]_. The
program is freely available at
http://theory.cm.utexas.edu/henkelman/research/bader/ where you will
also find a description of the method.

The algorithm is very well suited for large solid state physical
systems, as well as large biomolecular systems. The computational time
depends only on the size of the 3D grid used to represent the electron
density, and scales linearly with the total number of grid points. As
the accuracy of the method depends on the grid spacing, it is
recommended to check for convergence in this parameter (which should
usually by smaller than the default value).

The program takes cube input files. It does *not* support units, and
assumes atomic units for the density (`\text{bohr}^{-3}`).

A simple Python script for making a
cube file, ready for the Bader program, could be:

>>> from ase.io import write
>>> from ase.units import Bohr
>>> density = calc.get_all_electron_density() * Bohr**3
>>> write('density.cube', atoms, data=density)

One can also use :meth:`~gpaw.calculator.GPAW.get_pseudo_density` but it is
better to use the :meth:`~gpaw.calculator.GPAW.get_all_electron_density`
method as it will create a normalized electron density with all the electrons.

Note that it is strongly recommended to use version 0.26b or higher of
the program, and the examples below refer to this version.

.. seealso::

    :ref:`all electron density`


Example: The water molecule
---------------------------

The following example shows how to do Bader analysis for a water molecule.

First do a ground state calculation, and save the density as a cube file:

.. literalinclude:: h2o.py

Then analyse the density cube file by running (use *bader -h* for a
description of the possible options)::

  $ bader -p all_atom -p atom_index density.cube

This will produce a number of files. The *ACF.dat* file, contains a
summary of the Bader analysis:

.. literalinclude:: ACF.dat

Revealing that 0.58 electrons have been transferred from each
Hydrogen atom to the Oxygen atom.

The *BvAtxxxx.dat* files, are cube files for each Bader volume,
describing the density within that volume. (I.e. it is just the
original cube file, truncated to zero outside the domain of the
specific Bader volume).

*AtIndex.dat* is a cube file with an integer value at each grid point,
describing which Bader volume it belongs to.

The plot below shows the dividing surfaces of the Hydrogen Bader
volumes.  This was achieved by plotting a contour surface of
*AtIndex.dat* at an isovalue of 1.5 (see :download:`plot.py`).

.. image:: h2o-bader.png


.. [Bader] R. F. W. Bader.  Atoms in Molecules: A Quantum Theory.
           Oxford University Press, New York, 1990.

.. [Tang]  W. Tang, E. Sanville, G. Henkelman.
           A grid-based Bader analysis algorithm without lattice bias.
           J. Phys.: Compute Mater. 21, 084204 (2009).
