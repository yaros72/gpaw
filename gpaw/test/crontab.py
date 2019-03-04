"""Test GPAW.

Initial setup::

    cd ~
    python3 -m venv gpaw-tests
    cd gpaw-tests
    . bin/activate
    pip install scipy
    git clone http://gitlab.com/ase/ase.git
    cd ase
    pip install -U .
    cd ..
    git clone http://gitlab.com/gpaw/gpaw.git
    cd gpaw
    python setup.py install ...

Crontab::

    GPAW_COMPILE_OPTIONS="..."
    CMD="python3 -m gpaw.test.crontab"
    10 20 * * * cd ~/gpaw-tests; . bin/activate; $CMD > test.out

"""
from __future__ import print_function
import os
import subprocess
import sys


cmds = """\
pip install --upgrade pip
touch gpaw-tests.lock
cd ase; git pull -q; pip install -U .
cd gpaw; git clean -fdx; git pull -q
cd gpaw; python setup.py install {} 2> ../test.err
gpaw test > test-1.out
gpaw -P 2 test > test-2.out
gpaw -P 4 test > test-4.out
gpaw -P 8 test > test-8.out
grep -i fail test-?.out >&2 || echo OK"""
# grep return error code 1 if nothing found.  So we end with "echo OK" to get
# a zero error code.

cmds = cmds.format(os.environ.get('GPAW_COMPILE_OPTIONS', ''))

if os.path.isfile('gpaw-tests.lock'):
    sys.exit('Locked')
try:
    for cmd in cmds.splitlines():
        subprocess.check_call(cmd, shell=True)
finally:
    os.remove('gpaw-tests.lock')
