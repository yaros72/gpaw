# creates: lines.png
from __future__ import division
import datetime as dt
import os
import subprocess

import numpy as np
import pylab as pl


def count(dir, pattern):
    if not os.path.isdir(dir):
        return 0
    files = subprocess.check_output('find {} -name {}'.format(dir, pattern),
                                    shell=True).decode().split()[:-1]
    if not files:
        return 0
    out = subprocess.check_output('wc -l {} | tail -1'.format(' '.join(files)),
                                  shell=True)
    return int(out.split()[0])


def polygon(x, y1, y2, *args, **kwargs):
    x = pl.concatenate((x, x[::-1]))
    y = pl.concatenate((y1, y2[::-1]))
    pl.fill(x, y, *args, **kwargs)


def plot_count(dpi=70):
    year, month, f, c, py, test, doc, rst = np.loadtxt('lines.data').T
    date = year + (month - 1) / 12

    fig = pl.figure(1, figsize=(10, 5), dpi=dpi)
    fig.add_subplot(111)
    polygon(date, c + py + test + doc, c + py + test + doc + rst,
            facecolor='m', label='Documentation (.rst)')
    polygon(date, c + py + test, c + py + test + doc,
            facecolor='c', label='Documentation (.py)')
    polygon(date, c + py, c + py + test,
            facecolor='y', label='Tests (.py)')
    polygon(date, c, c + py,
            facecolor='g', label='Python-code (.py) ')
    polygon(date, f, c,
            facecolor='r', label='C-code (.c, .h)')
    polygon(date, f, f,
            facecolor='b', label='Fortran-code')

    pl.axis('tight')
    pl.legend(loc='upper left')
    pl.title('Number of lines')
    pl.savefig('lines.png', dpi=dpi)


def count_lines():
    now = dt.date.today()
    stop = now.year, now.month
    year = 2005
    month = 11

    fd = open('lines.data', 'w')
    while (year, month) <= stop:
        hash = subprocess.check_output(
            'git rev-list -n 1 --before="{}-{}-01 12:00" master'
            .format(year, month), shell=True).strip()
        print(year, month, hash)
        subprocess.call(['git', 'checkout', hash])

        c = count('c', '\*.[ch]')
        py = count('.', '\*.py')
        test = count('gpaw/test', '\*.py')
        test += count('test', '\*.py')
        doc = count('doc', '\*.py')
        py -= test + doc  # avoid double counting
        rst = count('.', '\*.rst')
        print(year, month, 0, c, py, test, doc, rst, file=fd)
        month += 1
        if month == 13:
            month = 1
            year += 1

    fd.close()


plot_count()
