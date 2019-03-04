.. program:: gpaw
.. highlight:: bash
.. index:: gpaw, command line interface, CLI

.. _cli:

======================
Command line interface
======================

GPAW has a command line tool called :program:`gpaw` with the following
sub-commands:

==============  =====================================================
sub-command     description
==============  =====================================================
help            Help for sub-command
run             Run calculation with GPAW
info            Show versions of GPAW and its dependencies
dos             Calculate (projected) density of states from gpw-file
gpw             Write summary of GPAW-restart file
completion      Add tab-completion for Bash
test            Run the GPAW test suite
atom            Solve radial equation for an atom
python          Run GPAW's parallel Python interpreter
sbatch          Submit a GPAW Python script via sbatch
dataset         Calculate density of states from gpw-file
symmetry        Analyse symmetry
install-data    Install PAW datasets, pseudopotential or basis sets
==============  =====================================================


Help
====

You can do::

    $ gpaw --help
    $ gpaw sub-command --help

to get help (or ``-h`` for short).


.. _bash completion:

Bash completion
===============

You can enable bash completion like this::

    $ gpaw completions

This will append a line like this::

    complete -o default -C /path/to/gpaw/gpaw/cli/complete.py gpaw

to your ``~/.bashrc``.
