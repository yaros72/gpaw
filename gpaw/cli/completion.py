import os
import sys

from ase.cli.completion import update, CLICommand

from gpaw.cli.main import commands

my_dir, _ = os.path.split(os.path.realpath(__file__))
script = os.path.join(my_dir, 'complete.py')

CLICommand.cmd = ('complete -o default -C "{py} {script}" gpaw'
                  .format(py=sys.executable, script=script))


if __name__ == '__main__':
    # Path of the complete.py script:
    update(script, commands)
