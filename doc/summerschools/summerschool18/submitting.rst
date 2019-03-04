.. _gbar submitting:

====================================
Submitting jobs on the DTU computers
====================================

Smaller calculations can be run in a Jupyter Notebook, but larger calculations
require running on multiple CPU cores for an extended time.  Such jobs should
be submitted with the ``qsub.py``.


Using ``qsub.py``
=================

The script ``qsub.py`` acts as a GPAW-aware front-end to the queue system.
Usage::

  qsub.py [-h] [-p PROCESSES] [-t TIME] [-z] script [argument [argument ...]]

Submit a GPAW Python script via qsub.

positional arguments:
  script:
    Python script

  argument:
    Command-line argument for Python script.

optional arguments:
  -h, --help            show this help message and exit
  -p PROCESSES, --processes PROCESSES
                        Number of processes.
  -t TIME, --time TIME  Max running time in hours.
  -z, --dry-run         Don't actually submit script.


.. code:: bash

    $ qsub.py -p 8 -t 4 script.py  # 8 cores, 4 hours
    $ qstat -u <username>
    ...


Choosing the number of processes
================================

GPAW parallelizes most efficiently over k-points, so it is a good idea to make
the number of processes a divisor of the number of *irreducible* k-points.  If
you have 12 irreducible k-points, the calculation parallelizes well on 2, 3,
4, 6 or 12 processes.

If you have very few irreducible k-points you may need to have more processes
than k-points; in these cases GPAW choose other parallelization strategies.
In this case, it is an advantage to make the number of processes a multiple of
the number of irreducible k-points.


Dry run: Let GPAW help you choosing
===================================

If you run your script with the command::

  python3 myscript.py --gpaw dry-run=1

then your script will execute until the first GPAW calculation.  That
calculation will print information into the ``.txt`` file, and then stop.  In
the file, you can see the number of irreducible k-points and use it to select
your parallelization strategy.

Once you have decided how many processes you want, run another dry-run to
check how GPAW will parallelize::

  python3 myscript.py --gpaw dry-run=PROCESSES

where ``PROCESSES`` is the number of processes you want to use.  In this case,
gpaw will print how it will parallelize the calculation when running for real.
