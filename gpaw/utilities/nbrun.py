"""Extract code from ipynb file.

Used for testing summerschool notebooks.
"""
import json
from pathlib import Path


def convert(path):
    data = json.loads(path.read_text())

    lines = ['# Converted from {path}\n'.format(path=path)]
    n = 1
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            lines.extend(['\n', '# In [{n}]:\n'.format(n=n)])
            for line in cell['source']:
                if line.startswith('%') or line.startswith('!'):
                    line = '# ' + line
                lines.append(line)
            lines.append('\n')
            n += 1

    code = ''.join(lines)

    path.with_suffix('.py').write_text(code)

    return code


def view(atoms, repeat=None):
    pass


def run(name):
    """Execute ipynb file.

    Run code with ase.visualize.view() disabled.
    """
    import ase.visualize as visualize
    visualize.view = view
    code = convert(Path(name))
    exec(code, {})


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
