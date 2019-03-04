# encoding: utf-8
# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

"""Main gpaw module."""

import sys

from gpaw.broadcast_imports import broadcast_imports

with broadcast_imports:
    import os
    import runpy
    import warnings
    from sysconfig import get_platform
    from os.path import join, isfile
    from argparse import ArgumentParser, REMAINDER, RawDescriptionHelpFormatter

    import numpy as np
    from ase.cli.run import str2dict


assert not np.version.version.startswith('1.6.0')

__version__ = '1.5.2b1'
__ase_version_required__ = '3.17.1b1'

__all__ = ['GPAW',
           'Mixer', 'MixerSum', 'MixerDif', 'MixerSum2',
           'CG', 'Davidson', 'RMMDIIS', 'DirectLCAO',
           'PoissonSolver',
           'FermiDirac', 'MethfesselPaxton',
           'PW', 'LCAO', 'restart', 'FD']


class ConvergenceError(Exception):
    pass


class KohnShamConvergenceError(ConvergenceError):
    pass


class PoissonConvergenceError(ConvergenceError):
    pass


class KPointError(Exception):
    pass


def parse_extra_parameters(arg):
    return {key.replace('-', '_'): value
            for key, value in str2dict(arg).items()}


is_gpaw_python = '_gpaw' in sys.builtin_module_names


def parse_arguments(argv):
    p = ArgumentParser(usage='%(prog)s [OPTION ...] [-c | -m] SCRIPT'
                       ' [ARG ...]',
                       description='Run a parallel GPAW calculation.\n\n'
                       'Compiled with:\n  Python {}'
                       .format(sys.version.replace('\n', '')),
                       formatter_class=RawDescriptionHelpFormatter)

    p.add_argument('--command', '-c', action='store_true',
                   help='execute Python string given as SCRIPT')
    p.add_argument('--module', '-m', action='store_true',
                   help='run library module given as SCRIPT')
    p.add_argument('-W', metavar='argument',
                   action='append', default=[], dest='warnings',
                   help='warning control.  See the documentation of -W for '
                   'the Python interpreter')
    p.add_argument('--memory-estimate-depth', default=2, type=int, metavar='N',
                   dest='memory_estimate_depth',
                   help='print memory estimate of object tree to N levels')
    p.add_argument('--domain-decomposition',
                   metavar='N or X,Y,Z', dest='parsize_domain',
                   help='use N or X × Y × Z cores for domain decomposition.')
    p.add_argument('--state-parallelization', metavar='N', type=int,
                   dest='parsize_bands',
                   help='use N cores for state/band/orbital parallelization')
    p.add_argument('--augment-grids', action='store_true',
                   dest='augment_grids',
                   help='when possible, redistribute real-space arrays on '
                   'cores otherwise used for k-point/band parallelization')
    p.add_argument('--buffer-size', type=float, metavar='SIZE',
                   help='buffer size for MatrixOperator in MiB')
    p.add_argument('--profile', metavar='FILE', dest='profile',
                   help='run profiler and save stats to FILE')
    p.add_argument('--gpaw', metavar='VAR=VALUE[, ...]', action='append',
                   default=[], dest='gpaw_extra_kwargs',
                   help='extra (hacky) GPAW keyword arguments')
    p.add_argument('--benchmark-imports', action='store_true',
                   help='count distributed/non-distributed imports')
    if is_gpaw_python:  # SCRIPT mandatory for gpaw-python
        p.add_argument('script', metavar='SCRIPT',
                       help='calculation script')
    p.add_argument('options', metavar='ARG',
                   help='arguments forwarded to SCRIPT', nargs=REMAINDER)

    args = p.parse_args(argv[1:])

    if args.command and args.module:
        p.error('-c and -m are mutually exclusive')

    if is_gpaw_python:
        sys.argv = [args.script] + args.options

    for w in args.warnings:
        # Need to convert between python -W syntax to call
        # warnings.filterwarnings():
        warn_args = w.split(':')
        assert len(warn_args) <= 5

        if warn_args[0] == 'all':
            warn_args[0] = 'always'
        if len(warn_args) >= 3:
            # e.g. 'UserWarning' (string) -> UserWarning (class)
            warn_args[2] = globals().get(warn_args[2])
        if len(warn_args) == 5:
            warn_args[4] = int(warn_args[4])

        warnings.filterwarnings(*warn_args, append=True)

    if args.parsize_domain:
        parsize = [int(n) for n in args.parsize_domain.split(',')]
        if len(parsize) == 1:
            parsize = parsize[0]
        else:
            assert len(parsize) == 3
        args.parsize_domain = parsize

    extra_parameters = {}
    for extra_kwarg in args.gpaw_extra_kwargs:
        extra_parameters.update(parse_extra_parameters(extra_kwarg))

    return extra_parameters, args


if is_gpaw_python:
    extra_parameters, gpaw_args = parse_arguments(sys.argv)
    # The normal Python interpreter puts . in sys.path, so we also do that:
    sys.path.insert(0, '.')
else:
    # Ignore the arguments; rely on --gpaw only as below.
    extra_parameters, gpaw_args = parse_arguments([sys.argv[0]])


def parse_gpaw_args():
    extra_parameters = {}
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith('--gpaw='):
            sys.argv.pop(i)
            extra_parameters.update(parse_extra_parameters(arg[7:]))
            continue
            break
        elif arg == '--gpaw':
            sys.argv.pop(i)
            extra_parameters.update(parse_extra_parameters(sys.argv.pop(i)))
            continue
            break
        i += 1
    return extra_parameters


extra_parameters.update(parse_gpaw_args())


# Check for special command line arguments:
memory_estimate_depth = gpaw_args.memory_estimate_depth
parsize_domain = gpaw_args.parsize_domain
parsize_bands = gpaw_args.parsize_bands
augment_grids = gpaw_args.augment_grids
# We deprecate the sl_xxx parameters being set from command line.
# People can satisfy their lusts by setting gpaw.sl_default = something
# if they are perverted enough to use global variables.
sl_default = None
sl_diagonalize = None
sl_inverse_cholesky = None
sl_lcao = None
sl_lrtddft = None
buffer_size = gpaw_args.buffer_size
profile = gpaw_args.profile


def main():
    # Stacktraces can be shortened by running script with
    # PyExec_AnyFile and friends.  Might be nicer
    if gpaw_args.command:
        d = {'__name__': '__main__'}
        exec(gpaw_args.script, d, d)
    elif gpaw_args.module:
        # Python has: python [-m MOD] [-c CMD] [SCRIPT]
        # We use a much better way: gpaw-python [-m | -c] SCRIPT
        runpy.run_module(gpaw_args.script, run_name='__main__')
    else:
        runpy.run_path(gpaw_args.script, run_name='__main__')

    # Todo, if we want: interactive interpreter.


dry_run = extra_parameters.pop('dry_run', 0)
debug = extra_parameters.pop('debug', False)

# Check for typos:
for p in extra_parameters:
    # We should get rid of most of these!
    if p not in {'sic', 'log2ng', 'PK', 'vdw0', 'df_dry_run', 'usenewlfc'}:
        warnings.warn('Unknown parameter: ' + p)

if debug:
    np.seterr(over='raise', divide='raise', invalid='raise', under='ignore')

    oldempty = np.empty

    def empty(*args, **kwargs):
        a = oldempty(*args, **kwargs)
        try:
            a.fill(np.nan)
        except ValueError:
            a.fill(-1000000)
        return a
    np.empty = empty


build_path = join(__path__[0], '..', 'build')
arch = '%s-%s' % (get_platform(), sys.version[0:3])

# If we are running the code from the source directory, then we will
# want to use the extension from the distutils build directory:
sys.path.insert(0, join(build_path, 'lib.' + arch))


def get_gpaw_python_path():
    paths = os.environ['PATH'].split(os.pathsep)
    paths.insert(0, join(build_path, 'bin.' + arch))
    for path in paths:
        if isfile(join(path, 'gpaw-python')):
            return path
    raise RuntimeError('Could not find gpaw-python!')


setup_paths = []


def initialize_data_paths():
    try:
        setup_paths[:] = os.environ['GPAW_SETUP_PATH'].split(os.pathsep)
    except KeyError:
        if os.pathsep == ';':
            setup_paths[:] = [r'C:\gpaw-setups']
        else:
            setup_paths[:] = ['/usr/local/share/gpaw-setups',
                              '/usr/share/gpaw-setups']


initialize_data_paths()

with broadcast_imports:
    from gpaw.calculator import GPAW
    from gpaw.mixer import Mixer, MixerSum, MixerDif, MixerSum2
    from gpaw.eigensolvers import Davidson, RMMDIIS, CG, DirectLCAO
    from gpaw.poisson import PoissonSolver
    from gpaw.occupations import FermiDirac, MethfesselPaxton
    from gpaw.wavefunctions.lcao import LCAO
    from gpaw.wavefunctions.pw import PW
    from gpaw.wavefunctions.fd import FD

RMM_DIIS = RMMDIIS


def restart(filename, Class=GPAW, **kwargs):
    calc = Class(filename, **kwargs)
    atoms = calc.get_atoms()
    return atoms, calc


if profile:
    from cProfile import Profile
    import atexit
    prof = Profile()

    def f(prof, filename):
        prof.disable()
        if filename == '-':
            prof.print_stats('time')
        else:
            from gpaw.mpi import rank
            prof.dump_stats(filename + '.%04d' % rank)
    atexit.register(f, prof, profile)
    prof.enable()


command = os.environ.get('GPAWSTARTUP')
if command is not None:
    exec(command)


def is_parallel_environment():
    """Check if we are running in a parallel environment.

    This function can be redefined in ~/.gpaw/rc.py.  Example::

        def is_parallel_environment():
            import os
            return 'PBS_NODEFILE' in os.environ
    """
    return False


def read_rc_file():
    home = os.environ.get('HOME')
    if home is not None:
        rc = os.path.join(home, '.gpaw', 'rc.py')
        if os.path.isfile(rc):
            # Read file in ~/.gpaw/rc.py
            with open(rc) as fd:
                exec(fd.read())


read_rc_file()
