.. _restart_files:

=============
Restart files
=============

Writing restart files
=====================

Use ``calc.write('xyz.gpw')`` or ``calc.write('xyz.gpw', mode='all')``
to include also the wave functions.

You can register an automatic call to the ``write`` method, every
``n``'th iteration like this::

  calc.attach(calc.write, n, 'xyz.gpw')

or::

  calc.attach(calc.write, n, 'xyz.gpw', mode='all')


Reading restart files
=====================

The calculation can be read from file like this::

  calc = GPAW('xyz.gpw')

or this::

  atoms, calc = restart('xyz.gpw')

By adding the option txt=None you can suppress text output when restarting
(e.g. when plotting a DOS)::

  atoms, calc = restart('xyz.gpw', txt=None)
