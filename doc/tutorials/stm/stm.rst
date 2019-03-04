.. _stm tutorial:

==============================
Tutorial: STM images - Al(111)
==============================

Let's make a 2 layer Al(111) fcc surface using the
:func:`ase.build.fcc111` function:

.. literalinclude:: al111.py
   :lines: 1-3

Now we calculate the wave functions and write them to a file:

.. literalinclude:: al111.py
   :lines: 5-


2-d scans
=========

First initialize the :class:`~ase.dft.stm.STM` object and get the
averaged current at `z=8.0` Å (for our surface, the top layer is at `z=6.338`
Å):

.. literalinclude:: stm.py
   :end-before: plot1

From the current we make a scan to get a 2-d array of constant current
height and make a contour plot:

.. literalinclude:: stm.py
   :start-after: plot1
   :end-before: plot2

.. image:: 2d.png

Similarly, we can make a constant height scan (at `z=8.0` Å) and plot it:

.. literalinclude:: stm.py
   :start-after: plot2
   :end-before: plot3

.. image:: 2d_I.png

Linescans
=========

Here is how to make a line-scan:

.. literalinclude:: stm.py
   :start-after: plot3
   :end-before: plot4

.. image:: line.png

Scanning tunneling spectroscopy
===============================

We can also make STS plots (dV/dV curve at specified location; here at `z=8.0` Å above atom 0:

.. literalinclude:: stm.py
   :start-after: plot4

.. image:: dIdV.png
