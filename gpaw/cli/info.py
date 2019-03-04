import os.path as op
import subprocess
import sys

from ase.utils import import_module
from ase.utils import search_current_git_hash

import gpaw
import _gpaw
import gpaw.fftw as fftw
from gpaw.mpi import rank, have_mpi
from gpaw.utilities import compiled_with_sl, compiled_with_libvdwxc
from gpaw.utilities.elpa import LibElpa


def info():
    """Show versions of GPAW and its dependencies."""
    results = [('python-' + sys.version.split()[0], sys.executable)]
    for name in ['gpaw', 'ase', 'numpy', 'scipy']:
        try:
            module = import_module(name)
        except ImportError:
            results.append((name, False))
        else:
            # Search for git hash
            githash = search_current_git_hash(module)
            if githash is None:
                githash = ''
            else:
                githash = '-{:.10}'.format(githash)
            results.append((name + '-' + module.__version__ + githash,
                            module.__file__.rsplit('/', 1)[0] + '/'))
    results.append(('libxc-' + _gpaw.libxc_version, ''))
    module = import_module('_gpaw')
    if hasattr(module, 'githash'):
        githash = '-{:.10}'.format(module.githash())
    results.append(('_gpaw' + githash,
                    op.normpath(getattr(module, '__file__', 'built-in'))))
    p = subprocess.Popen(['which', 'gpaw-python'], stdout=subprocess.PIPE)
    results.append(('parallel', p.communicate()[0].strip().decode() or False))
    results.append(('MPI enabled', have_mpi))
    if have_mpi:
        have_sl = compiled_with_sl()
        have_elpa = LibElpa.have_elpa()
    else:
        have_sl = have_elpa = 'no (MPI unavailable)'
    results.append(('scalapack', have_sl))
    results.append(('Elpa', have_elpa))
    results.append(('FFTW', fftw.FFTPlan is fftw.FFTWPlan))
    results.append(('libvdwxc', compiled_with_libvdwxc()))
    paths = ['{0}: {1}'.format(i + 1, path)
             for i, path in enumerate(gpaw.setup_paths)]
    results.append(('PAW-datasets', '\n{:25}'.format('').join(paths)))

    if rank == 0:
        for a, b in results:
            if isinstance(b, bool):
                b = ['no', 'yes'][b]
            print('{0:25}{1}'.format(a, b))


class CLICommand:
    __doc__ = info.__doc__.rstrip('.')

    @staticmethod
    def add_arguments(parser):
        pass

    @staticmethod
    def run(args):
        info()
        if not have_mpi:
            print()
            print('MPI not enabled.  Check parallel configuration with: '
                  'gpaw -P1 info')
