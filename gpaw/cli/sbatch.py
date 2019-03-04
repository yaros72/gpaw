from __future__ import print_function
import subprocess
import sys

usage = """gpaw sbatch [-0] -- [sbatch options] script.py [script options]
   or: gpaw sbatch [-0] -- [sbatch options] python -m module [module options]
"""


class CLICommand:
    """Submit a GPAW Python script via sbatch."""

    @staticmethod
    def add_arguments(parser):
        parser.usage = usage
        parser.add_argument('-0', '--dry-run', action='store_true')
        parser.add_argument('arguments', nargs='*')

    @staticmethod
    def run(args):
        script = '#!/bin/bash -l\n'
        for i, arg in enumerate(args.arguments):
            if arg.endswith('.py'):
                break
        else:
            for i, arg in enumerate(args.arguments):
                if (arg.startswith('python') and
                    len(args.arguments) > i + 1 and
                    args.arguments[i + 1].startswith('-m')):
                    del args.arguments[i]
                    break
            else:
                print('No script.py found!', file=sys.stderr)
                return

        if arg.endswith('.py'):
            for line in open(arg):
                if line.startswith('#SBATCH'):
                    script += line
        script += ('cd $SLURM_SUBMIT_DIR\n')
        script += ('OMP_NUM_THREADS=1 '
                   'mpiexec `echo $GPAW_MPI_OPTIONS` gpaw-python {}\n'
                   .format(' '.join(args.arguments[i:])))
        cmd = ['sbatch', '--export=NONE'] + args.arguments[:i]
        if args.dry_run:
            print('sbatch command:')
            print(' '.join(cmd))
            print('\nscript:')
            print(script, end='')
        else:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            p.communicate(script.encode())
