# -*- coding: utf-8 -*-
# creates: H.rst, H.default.png
# ... and all the rest.
from __future__ import print_function
import json
import sys

import matplotlib.pyplot as plt
from ase.data import atomic_numbers, atomic_names
from ase.units import Hartree
from ase.utils import plural

from gpaw.atom.check import cutoffs, all_names


with open('datasets.json') as fd:
    data = json.load(fd)


def rst(symbol):
    Z = atomic_numbers[symbol]
    name = atomic_names[Z]

    rst = """\
.. Computer generated reST (make_setup_pages.py)
.. index:: {name}
.. _{name}:

================
{name}
================

Datasets:

.. csv-table::
    :header: name, valence electrons, frozen core electrons

{table}"""

    table = ''
    for e, nlfer, energies in data[symbol]:
        nv, txt = rst1(symbol + '.' + e, nlfer, energies)
        if e != 'default':
            e = "``'{}'``".format(e)
        table += '    {},{},{}\n'.format(e, nv, Z - nv)
        rst += txt

    with open(symbol + '.rst', 'w') as fd:
        fd.write(rst.format(table=table, name=name))


def rst1(dataset, nlfer, energies):
    table1 = ''
    nv = 0
    for n, l, f, e, rcut in nlfer:
        n, l, f = (int(x) for x in [n, l, f])
        if n == -1:
            n = ''
        table1 += '    {}{},{},{:.3f},'.format(n, 'spdf'[l], f, e * Hartree)
        if rcut:
            table1 += '{:.2f}'.format(rcut)
            nv += f
        table1 += '\n'

    rst = """

{electrons}
====================

Radial cutoffs and eigenvalues:

.. csv-table::
    :header: id, occ, eig [eV], cutoff [Bohr]

{table1}

The figure shows convergence of the absolute energy (red line)
and atomization energy (green line) of a {symbol} dimer relative
to completely converged numbers (plane-wave calculation at 1500 eV).
Also shown are finite-difference and LCAO (dzp) calculations at gridspacings
0.143 Å, 0.167 Å and 0.200 Å.

.. image:: {dataset}.png

Egg-box errors in finite-difference mode:

.. csv-table::
    :header: grid-spacing [Å], energy error [eV]

{table2}"""

    epw, depw, efd, defd, elcao, delcao, deegg = energies

    table2 = ''
    for h, e in zip([0.16, 0.18, 0.2], deegg):
        table2 += '    {:.2f},{:.4f}\n'.format(h, e)

    fig = plt.figure(figsize=(8, 5))

    ax1 = plt.subplot(121)
    ax1.semilogy(cutoffs[:-1], epw[:-1], 'r',
                 label='pw, absolute')
    ax1.semilogy(cutoffs[:-1], depw[:-1], 'g',
                 label='pw, atomization')
    plt.xticks([200, 400, 600])
    plt.xlabel('plane-wave cutoff [eV]')
    plt.ylabel('error [eV/atom]')
    plt.legend(loc='best')

    ax2 = plt.subplot(122, sharey=ax1)
    h = [4.0 / g for g in [20, 24, 28]]
    ax2.semilogy(h, efd, '-rs', label='fd, absolute')
    ax2.semilogy(h, defd, '-gs', label='fd, atomization')
    ax2.semilogy(h, elcao, '-ro', label='lcao, absolute')
    ax2.semilogy(h, delcao, '-go', label='lcao, atomization')
    plt.xticks([0.16, 0.18, 0.2])
    plt.xlim(0.14, 0.2)
    plt.xlabel(u'grid-spacing [Å]')
    plt.legend(loc='best')
    plt.setp(ax2.get_yticklabels(), visible=False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    plt.savefig(dataset + '.png')
    plt.close(fig)

    return nv, rst.format(electrons=plural(nv, 'valence electron'),
                          table1=table1, table2=table2, symbol=symbol,
                          dataset=dataset)


for symbol in all_names:
    if '.' not in symbol:
        print(symbol, end='')
        sys.stdout.flush()
        rst(symbol)
print()
