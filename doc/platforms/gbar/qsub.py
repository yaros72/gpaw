#!/usr/bin/env python3
"""GPAW-wrapper for qsub."""

import argparse
import subprocess

description = 'Submit a GPAW Python script via qsub.'
maxcores = 8  # XeonX5550


def main():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-p', '--processes', type=int, default=4,
                        help='Number of processes.')
    parser.add_argument('-t', '--time', type=int, default=1,
                        help='Max running time in hours.')
    parser.add_argument('-z', '--dry-run', action='store_true',
                        help='Don\'t actually submit script.')
    parser.add_argument('script', help='Python script')
    parser.add_argument('argument', nargs='*',
                        help='Command-line argument for Python script.')
    args = parser.parse_args()
    arguments = ' '.join(args.argument)
    cmd = f'gpaw-python {args.script} {arguments}'
    nodes, rest = divmod(args.processes, maxcores)
    if nodes > 0:
        ppn = maxcores
        if rest > 0:
            msg = f'Please use a multiple of {maxcores} processes!'
            raise SystemExit(msg)
    else:
        nodes = 1
        ppn = args.processes
    lines = [
        '#!/bin/sh',
        '#PBS -q hpc',
        f'#PBS -N {args.script}',
        f'#PBS -l nodes={nodes}:ppn={ppn}',
        f'#PBS -l walltime={args.time}:00:00',
        # '#PBS -l feature=XeonX5550',
        'cd $PBS_O_WORKDIR',
        f'OMP_NUM_THREADS=1 mpiexec {cmd}']
    script = '\n'.join(lines) + '\n'
    if args.dry_run:
        print(script)
    else:
        p = subprocess.Popen(['qsub'], stdin=subprocess.PIPE)
        p.communicate(script.encode())


if __name__ == '__main__':
    main()
