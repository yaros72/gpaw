#!/usr/bin/env python3
"""Bash completion for ase.

Put this in your .bashrc::

    complete -o default -C /path/to/gpaw/cli/complete.py gpaw

or run::

    $ gpaw completion [-0] [<your-bash-file>]

"""

from __future__ import print_function
import os
import sys
from glob import glob


def match(word, *suffixes):
    return [w for w in glob(word + '*')
            if any(w.endswith(suffix) for suffix in suffixes)]


# Beginning of computer generated data:
commands = {
    'atom':
        ['-f', '--xc-functional', '-a', '--add', '--spin-polarized', '-d',
         '--dirac', '-p', '--plot', '-e', '--exponents', '-l',
         '--logarithmic-derivatives', '-r', '--refine', '-s',
         '--scalar-relativistic', '--no-ee-interaction'],
    'completion':
        [],
    'dataset':
        ['-f', '--xc-functional', '-C', '--configuration', '-P',
         '--projectors', '-r', '--radius', '-0',
         '--zero-potential', '-c',
         '--pseudo-core-density-radius', '-z', '--pseudize',
         '-p', '--plot', '-l', '--logarithmic-derivatives', '-w',
         '--write', '-s', '--scalar-relativistic', '-n',
         '--no-check', '-t', '--tag', '-a', '--alpha', '-g',
         '--gamma', '-b', '--create-basis-set', '--nlcc',
         '--core-hole', '-e', '--electrons', '-o', '--output'],
    'diag':
        ['-b', '--bands', '-s', '--scalapack', '-0', '--dry-run'],
    'dos':
        ['-p', '--plot', '-i', '--integrated', '-w', '--width', '-a',
         '--atom', '-t', '--total', '-r', '--range', '-n',
         '--points'],
    'gpw':
        ['-w', '--remove-wave-functions'],
    'info':
        [],
    'install-data':
        ['--version', '--tarball', '--list-all', '--gpaw', '--sg15',
         '--basis', '--test', '--register', '--no-register'],
    'python':
        [],
    'rpa':
        ['-e', '--cut-off', '-b', '--blocks'],
    'run':
        ['-p', '--parameters', '-t', '--tag', '--properties', '-f',
         '--maximum-force', '--constrain-tags', '-s',
         '--maximum-stress', '-E', '--equation-of-state',
         '--eos-type', '--modify', '--after', '-i',
         '--interactive', '-c', '--collection', '-d',
         '--database', '-S', '--skip', '-w', '--write', '-W',
         '--write-all'],
    'sbatch':
        ['-0', '--dry-run'],
    'symmetry':
        [],
    'test':
        ['-x', '--exclude', '-f', '--run-failed-tests-only', '--from',
         '--after', '--range', '-j', '--jobs', '--reverse', '-k',
         '--keep-temp-dir', '-d', '--directory', '-s',
         '--show-output', '--list']}
# End of computer generated data


def complete(word, previous, line, point):
    for w in line[:point - len(word)].strip().split()[1:]:
        if w[0].isalpha():
            if w in commands:
                command = w
                break
    else:
        if word[:1] == '-':
            return ['-h', '--help', '--version', '-P', '--parallel']
        return list(commands.keys()) + ['-h', '--help', '--version',
                                        '-P', '--parallel']

    if word[:1] == '-':
        return commands[command]

    words = []

    if command == 'help':
        words = commands

    elif command == 'test':
        from glob import glob
        path = __file__.rsplit('/', 1)[0] + '/../test/**/*.py'
        return (word.rsplit('/test/')[-1]
                for word in glob(path, recursive=True))

    return words


word, previous = sys.argv[2:]
line = os.environ['COMP_LINE']
point = int(os.environ['COMP_POINT'])
words = complete(word, previous, line, point)
for w in words:
    if w.startswith(word):
        print(w)
