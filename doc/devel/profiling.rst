.. _profiling:

=========
Profiling
=========

profile
=======

Python has a ``profile`` module to help you find the places in the
code where the time is spent.

Let's say you have a script ``script.py`` that you want to run through the
profiler.  This is what you do:

>>> import profile
>>> profile.run('import script', 'prof')

This will run your script and generate a profile in the file ``prof``.
You can also generate the profile by inserting a line like this in
your script::

  ...
  import profile
  profile.run('atoms.get_potential_energy()', 'prof')
  ...

To analyse the results, you do this::

 >>> import pstats
 >>> pstats.Stats('prof').strip_dirs().sort_stats('time').print_stats(20)
 Tue Oct 14 19:08:54 2008    prof

         1093215 function calls (1091618 primitive calls) in 37.430 CPU seconds

   Ordered by: internal time
   List reduced from 1318 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    37074   10.310    0.000   10.310    0.000 :0(calculate_spinpaired)
     1659    4.780    0.003    4.780    0.003 :0(relax)
   167331    3.990    0.000    3.990    0.000 :0(dot)
     7559    3.440    0.000    3.440    0.000 :0(apply)
      370    2.730    0.007   17.090    0.046 xc_correction.py:130(calculate_energy_and_derivatives)
    37000    0.780    0.000    9.650    0.000 xc_functional.py:657(get_energy_and_potential_spinpaired)
    37074    0.720    0.000   12.990    0.000 xc_functional.py:346(calculate_spinpaired)
      ...
      ...

The list shows the 20 functions where the most time is spent.  Check
the pstats_ documentation if you want to do more fancy things.

.. _pstats: http://docs.python.org/library/profile.html


.. tip::

   Since the ``profile`` module does not time calls to C-code, it
   is a good idea to run the code in debug mode - this will wrap
   calls to C-code in Python functions::

     $ python3 script.py --debug

.. tip::

   There is also a quick and simple way to profile a script::

     $ pyhton script.py --profile=prof

   This will produce a file called ``prof.0000`` where ``0000`` is the
   rank number (if your run the script in parallel, there will be one
   file per rank).

   Use::

     $ python3 script.py --profile=-

   to write the report directly to standard output.
