.. _testing:

============
Testing GPAW
============

Testing of gpaw is done by a nightly test suite consisting of many
small and quick tests and by a weekly set of larger test.


Quick test suite
================

Use the :program:`gpaw` command to run the tests::

    $ gpaw test --help
    usage: gpaw test [-h] [-x test1.py,test2.py,...] [-f] [--from TESTFILE]
                     [--after TESTFILE] [--range test_i.py,test_j.py] [-j JOBS]
                     [--reverse] [-k] [-d DIRECTORY] [-s] [--list]
                     [tests [tests ...]]

    Run the GPAW test suite. The test suite can be run in parallel with MPI
    through gpaw-python. The test suite supports 1, 2, 4 or 8 CPUs although some
    tests are skipped for some parallelizations. If no TESTs are given, run all
    tests supporting the parallelization.

    positional arguments:
      tests

    optional arguments:
      -h, --help            show this help message and exit
      -x test1.py,test2.py,..., --exclude test1.py,test2.py,...
                            Exclude tests (comma separated list of tests).
      -f, --run-failed-tests-only
                            Run failed tests only.
      --from TESTFILE       Run remaining tests, starting from TESTFILE
      --after TESTFILE      Run remaining tests, starting after TESTFILE
      --range test_i.py,test_j.py
                            Run tests in range test_i.py to test_j.py (inclusive)
      -j JOBS, --jobs JOBS  Run JOBS threads. Each test will be executed in serial
                            by one thread. This option cannot be used for
                            parallelization together with MPI.
      --reverse             Run tests in reverse order (less overhead with
                            multiple jobs)
      -k, --keep-temp-dir   Do not delete temporary files.
      -d DIRECTORY, --directory DIRECTORY
                            Run test in this directory
      -s, --show-output     Show standard output from tests.
      --list                list the full list of tests, then exit

A temporary directory will be made and the tests will run in that
directory.  If all tests pass, the directory is removed.

The test suite consists of a large number of small and quick tests
found in the :git:`gpaw/test/` directory.  The tests run nightly in serial
and in parallel.



Adding new tests
----------------

A test script should fulfill a number of requirements:

* It should be quick.  Preferably a few seconds, but a few minutes is
  OK.  If the test takes several minutes or more, consider making the
  test a :ref:`big test <big-test>`.

* It should not depend on other scripts.

* It should be possible to run it on 1, 2, 4, and 8 cores.

A test can produce standard output and files - it doesn't have to
clean up.  Remember to add the new test to list of all tests specified
in the :git:`gpaw/test/__init__.py` file.

Use this function to check results:

.. function:: gpaw.test.equal(x, y, tolerance=0, fail=True, msg='')


.. _big-test:
.. _agts:

Big tests
=========

The directory in :git:`gpaw/test/big/` contains a set of longer and more
realistic tests that we run every weekend.  These are submitted to a
queueing system of a large computer.


Adding new tests
----------------

To add a new test, create a script somewhere in the file hierarchy ending with
``agts.py`` (e.g. ``submit.agts.py`` or just ``agts.py``). ``AGTS`` is short
for Advanced GPAW Test System (or Another Great Time Sink). This script
defines how a number of scripts should be submitted to niflheim and how they
depend on each other. Consider an example where one script, ``calculate.py``,
calculates something and saves a ``.gpw`` file and another script,
``analyse.py``, analyses this output. Then the submit script should look
something like::

    def create_tasks():
        from myqueue.task import task
        return [task('calculate.py', cores=8, tmax='25m'),
                task('analyse.py', cores=1, tmax='5m',
                     deps=['calculate.py'])]

As shown, this script has to contain the definition of the function
``create_tasks()``.  See https://myqueue.readthedocs.io/ for more details.
