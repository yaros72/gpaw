"""Build GPAW's web-page.

Initial setup::

    cd ~
    python3 -m venv gpaw-web-page
    cd gpaw-web-page
    . bin/activate
    pip install sphinx-rtd-theme
    pip install Sphinx
    pip install matplotlib scipy
    git clone http://gitlab.com/ase/ase.git
    cd ase
    pip install -U .
    cd ..
    git clone http://gitlab.com/gpaw/gpaw.git
    cd gpaw
    python setup.py install

Crontab::

    WEB_PAGE_FOLDER=...
    CMD="python -m gpaw.utilities.build_web_page"
    10 20 * * * cd ~/gpaw-web-page; . bin/activate; cd gpaw; $CMD > ../gpaw.log

"""
from __future__ import print_function
import os
import subprocess
import sys

from gpaw import __version__


cmds = """\
touch ../gpaw-web-page.lock
cd ../ase; git checkout web-page -q; git pull -q; pip install .
git clean -fdx
git checkout web-page -q > /dev/null 2>&1
git pull -q > /dev/null 2>&1
python setup.py install
cd doc; sphinx-build -b html -d build/doctrees . build/html
mv doc/build/html gpaw-web-page
cd gpaw-web-page/_sources/setups; cp setups.rst.txt setups.txt
cd ../ase; git checkout master -q; pip install .
git clean -fdx doc
rm -r build
git checkout master -q > /dev/null 2>&1
git pull -q > /dev/null 2>&1
python setup.py install
cd doc; sphinx-build -b html -d build/doctrees . build/html
mv doc/build/html gpaw-web-page/dev
python setup.py sdist
cp dist/gpaw-*.tar.gz gpaw-web-page/
cp dist/gpaw-*.tar.gz gpaw-web-page/dev/
find gpaw-web-page -name install.html | xargs sed -i s/snapshot.tar.gz/{0}/g
tar -czf gpaw-web-page.tar.gz gpaw-web-page
cp gpaw-web-page.tar.gz {1}/tmp-gpaw-web-page.tar.gz
mv {1}/tmp-gpaw-web-page.tar.gz {1}/gpaw-web-page.tar.gz"""

cmds = cmds.format('gpaw-' + __version__ + '.tar.gz',
                   os.environ['WEB_PAGE_FOLDER'])


def build():
    if os.path.isfile('../gpaw-web-page.lock'):
        print('Locked', file=sys.stderr)
        return
    try:
        for cmd in cmds.splitlines():
            subprocess.check_call(cmd, shell=True)
    finally:
        os.remove('../gpaw-web-page.lock')


if __name__ == '__main__':
    build()
