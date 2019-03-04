from distutils.version import LooseVersion
from gpaw import __ase_version_required__
from ase import __version__

assert LooseVersion(__version__) >= __ase_version_required__
