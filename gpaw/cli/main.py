"""GPAW command-line tool."""
from __future__ import print_function
import os
import sys


from ase.cli.main import main as ase_main

from gpaw import __version__


commands = [
    ('run', 'gpaw.cli.run'),
    ('info', 'gpaw.cli.info'),
    ('dos', 'gpaw.cli.dos'),
    ('gpw', 'gpaw.cli.gpw'),
    ('completion', 'gpaw.cli.completion'),
    ('test', 'gpaw.test.test'),
    ('atom', 'gpaw.atom.aeatom'),
    ('diag', 'gpaw.fulldiag'),
    # ('quick', 'gpaw.cli.quick'),
    ('python', 'gpaw.cli.python'),
    ('sbatch', 'gpaw.cli.sbatch'),
    ('dataset', 'gpaw.atom.generator2'),
    ('symmetry', 'gpaw.symmetry'),
    ('rpa', 'gpaw.xc.rpa'),
    ('install-data', 'gpaw.cli.install_data')]


def hook(parser, args):
    parser.add_argument('-P', '--parallel', type=int, metavar='N',
                        help="Run on N CPUs.")
    args = parser.parse_args()

    if args.parallel:
        from gpaw.mpi import have_mpi
        if not have_mpi:
            # Start again using gpaw-python in parallel:
            arguments = ['mpiexec', '-np', str(args.parallel),
                         'gpaw-python']
            if args.command == 'python':
                arguments += args.arguments
            else:
                arguments += ['-m', 'gpaw'] + sys.argv[1:]
            os.execvp('mpiexec', arguments)

    return args


def main():
    ase_main('gpaw', 'GPAW command-line tool', __version__, commands, hook)
