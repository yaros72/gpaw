from __future__ import print_function
import sys
import xml.sax

import numpy as np

from gpaw.setup_data import search_for_file
from gpaw.atom.radialgd import radial_grid_descriptor


_basis_letter2number = {'s': 1, 'd': 2, 't': 3, 'q': 4}
_basis_number2letter = 'Xsdtq56789'


def parse_basis_name(name):
    """Parse any basis type identifier: 'sz', 'dzp', 'qztp', '4z3p', ... """

    zetacount = _basis_letter2number.get(name[0])
    if zetacount is None:
        zetacount = int(name[0])
    assert name[1] == 'z'

    if len(name) == 2:
        polcount = 0
    elif len(name) == 3:
        assert name[-1] == 'p'
        polcount = 1
    else:
        assert len(name) == 4 and name[-1] == 'p'
        polcount = _basis_letter2number.get(name[2])
        if polcount is None:
            polcount = int(name[2])

    return zetacount, polcount


def get_basis_name(zetacount, polarizationcount):
    zetachar = _basis_number2letter[zetacount]
    if polarizationcount == 0:
        return '%sz' % zetachar
    elif polarizationcount == 1:
        return '%szp' % zetachar
    else:
        polarizationchar = _basis_number2letter[polarizationcount]
        return '%sz%sp' % (zetachar, polarizationchar)


class Basis:
    def __init__(self, symbol, name, readxml=True, rgd=None, world=None):
        self.symbol = symbol
        self.name = name
        self.rgd = rgd
        self.bf_j = []
        self.generatorattrs = {}
        self.generatordata = ''
        self.filename = None

        if readxml:
            self.read_xml(world=world)

    def nao(self):  # implement as a property so we don't have to
        # catch all the places where Basis objects are modified without
        # updating it.  (we can do that later)
        return sum([2 * bf.l + 1 for bf in self.bf_j])
    nao = property(nao)

    def append(self, bf):
        self.bf_j.append(bf)

    def get_grid_descriptor(self):
        return self.rgd

    def tosplines(self):
        return [self.rgd.spline(bf.phit_g, bf.rc, bf.l, points=400)
                for bf in self.bf_j]

    def read_xml(self, filename=None, world=None):
        parser = BasisSetXMLParser(self)
        parser.parse(filename, world=world)

    def write_xml(self):
        """Write basis functions to file.

        Writes all basis functions in the given list of basis functions
        to the file "<symbol>.<name>.basis".
        """
        if self.name is None:
            filename = '%s.basis' % self.symbol
        else:
            filename = '%s.%s.basis' % (self.symbol, self.name)
        write = open(filename, 'w').write
        write('<paw_basis version="0.1">\n')

        generatorattrs = ' '.join(['%s="%s"' % (key, value)
                                   for key, value
                                   in self.generatorattrs.items()])
        write('  <generator %s>' % generatorattrs)
        for line in self.generatordata.split('\n'):
            write('\n    ' + line)
        write('\n  </generator>\n')

        write('  ' + self.rgd.xml())

        for bf in self.bf_j:
            write(bf.xml(indentation='  '))

        write('</paw_basis>\n')

    def reduce(self, name):
        """Reduce the number of basis functions.

        Example: basis.reduce('sz') will remove all non single-zeta
        and polarization functions."""

        zeta, pol = parse_basis_name(name)
        newbf_j = []
        N = {}
        p = 0
        for bf in self.bf_j:
            if 'polarization' in bf.type:
                if p < pol:
                    newbf_j.append(bf)
                    p += 1
            else:
                nl = (int(bf.type[0]), 'spdf'.index(bf.type[1]))
                if nl not in N:
                    N[nl] = 0
                if N[nl] < zeta:
                    newbf_j.append(bf)
                    N[nl] += 1
        self.bf_j = newbf_j

    def get_description(self):
        title = 'LCAO basis set for %s:' % self.symbol
        if self.name is not None:
            name = 'Name: ' + self.name
        else:
            name = 'This basis set does not have a name'
        if self.filename is None:
            fileinfo = 'This basis set was not loaded from a file'
        else:
            fileinfo = 'File: ' + self.filename
        nj = len(self.bf_j)
        count1 = 'Number of radial functions: %d' % nj
        count2 = 'Number of spherical harmonics: %d' % self.nao

        bf_lines = []
        for bf in self.bf_j:
            line = '  l=%d, rc=%.4f Bohr: %s' % (bf.l, bf.rc, bf.type)
            bf_lines.append(line)

        lines = [title, name, fileinfo, count1, count2]
        lines.extend(bf_lines)
        return '\n  '.join(lines)


class BasisFunction:
    """Encapsulates various basis function data."""
    def __init__(self, n=None, l=None, rc=None, phit_g=None, type=''):
        self.n = n
        self.l = l
        self.rc = rc
        self.phit_g = phit_g
        self.type = type
        if n is None or n < 0:
            self.name = 'l=%d %s' % (l, type)
        else:
            self.name = '%d%s %s' % (n, 'spdf'[l], type)

    def __repr__(self, gridid=None):
        txt = '<basis_function '
        if self.n is not None:
            txt += 'n="%d" ' % self.n
        txt += ('l="%r" rc="%r" type="%s"' % (self.l, self.rc, self.type))
        if gridid is not None:
            txt += ' grid="%s"' % gridid
        return txt + '>'

    def xml(self, gridid='grid1', indentation=''):
        txt = indentation + self.__repr__(gridid) + '\n'
        txt += indentation + '  ' + ' '.join(repr(x) for x in self.phit_g)
        txt += '\n' + indentation + '</basis_function>\n'
        return txt


class BasisSetXMLParser(xml.sax.handler.ContentHandler):
    def __init__(self, basis):
        xml.sax.handler.ContentHandler.__init__(self)
        self.basis = basis
        self.type = None
        self.rc = None
        self.data = None
        self.l = None

    def parse(self, filename=None, world=None):
        """Read from symbol.name.basis file.

        Example of filename: N.dzp.basis.  Use sz(dzp) to read
        the sz-part from the N.dzp.basis file."""

        basis = self.basis
        if '(' in basis.name:
            assert basis.name.endswith(')')
            reduced, name = basis.name.split('(')
            name = name[:-1]
        else:
            name = basis.name
            reduced = None
        fullname = '%s.%s.basis' % (basis.symbol, name)
        if filename is None:
            basis.filename, source = search_for_file(fullname, world=world)
        else:
            basis.filename = filename
            source = open(filename, 'rb').read()

        self.data = None
        xml.sax.parseString(source, self)

        if reduced:
            basis.reduce(reduced)

    def startElement(self, name, attrs):
        if sys.version_info[0] < 3:
            attrs.__contains__ = attrs.has_key

        basis = self.basis
        if name == 'paw_basis':
            basis.version = attrs['version']
        elif name == 'generator':
            basis.generatorattrs = dict(attrs)
            self.data = []
        elif name == 'radial_grid':
            basis.rgd = radial_grid_descriptor(**attrs)
        elif name == 'basis_function':
            self.l = int(attrs['l'])
            self.rc = float(attrs['rc'])
            self.type = attrs.get('type')
            self.data = []
            if 'n' in attrs:
                self.n = int(attrs['n'])
            elif self.type[0].isdigit():
                self.n = int(self.type[0])
            else:
                self.n = None

    def characters(self, data):
        if self.data is not None:
            self.data.append(data)

    def endElement(self, name):
        basis = self.basis
        if name == 'basis_function':
            phit_g = np.array([float(x) for x in ''.join(self.data).split()])
            bf = BasisFunction(self.n, self.l, self.rc, phit_g, self.type)
            basis.append(bf)
        elif name == 'generator':
            basis.generatordata = ''.join([line for line in self.data])


class BasisPlotter:
    def __init__(self, premultiply=True, normalize=False,
                 show=False, save=False, ext='png'):
        self.premultiply = premultiply
        self.show = show
        self.save = save
        self.ext = ext
        self.default_filename = '%(symbol)s.%(name)s.' + ext

        self.title = 'Basis functions: %(symbol)s %(name)s'
        self.xlabel = r'r [Bohr]'
        if premultiply:
            ylabel = r'$\tilde{\phi} r$'
        else:
            ylabel = r'$\tilde{\phi}$'
        self.ylabel = ylabel

        self.normalize = normalize

    def plot(self, basis, filename=None, **plot_args):
        import pylab as pl  # Should not import in module namespace
        if plot_args is None:
            plot_args = {}
        r_g = basis.rgd.r_g

        print('Element  :', basis.symbol)
        print('Name     :', basis.name)
        print()
        print('Basis functions')
        print('---------------')

        norm_j = []
        for j, bf in enumerate(basis.bf_j):
            ng = len(bf.phit_g)
            rphit_g = r_g[:ng] * bf.phit_g
            norm = (rphit_g**2 * basis.rgd.dr_g).sum()
            norm_j.append(norm)
            print(bf.type, '[norm=%0.4f]' % norm)

        print()
        print('Generator')
        for key, item in basis.generatorattrs.items():
            print('   ', key, ':', item)
        print()
        print('Generator data')
        print(basis.generatordata)

        if self.premultiply:
            factor = r_g
        else:
            factor = np.ones_like(r_g)

        dashes_l = [(), (6, 3), (4, 1, 1, 1), (1, 1)]

        pl.figure()
        for norm, bf in zip(norm_j, basis.bf_j):
            ng = len(bf.phit_g)
            y_g = bf.phit_g * factor[:ng]
            if self.normalize:
                y_g /= norm
            pl.plot(r_g[:ng], y_g, label=bf.type[:12],
                    dashes=dashes_l[bf.l], lw=2,
                    **plot_args)
        axis = pl.axis()
        rc = max([bf.rc for bf in basis.bf_j])
        newaxis = [0., rc, axis[2], axis[3]]
        pl.axis(newaxis)
        pl.legend()
        pl.title(self.title % basis.__dict__)
        pl.xlabel(self.xlabel)
        pl.ylabel(self.ylabel)

        if filename is None:
            filename = self.default_filename
        if self.save:
            pl.savefig(filename % basis.__dict__)

        if self.show:
            pl.show()
