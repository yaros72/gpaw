from __future__ import print_function
import sys
from optparse import OptionParser

from gpaw.atom.basis import BasisMaker
from gpaw.atom.configurations import parameters, parameters_extra
from gpaw.setup_data import SetupData
from gpaw.mpi import world

# Module for generating basis sets more suitable for excited states.
#
# Generates basis sets that include empty p orbitals as valence states
# for metals.  This improves the description of excited states and is
# useful for TDDFT, band structures (the part above the Fermi level),
# and more.


class BasisSpecification:
    def __init__(self, setup, jvalues, jextra):
        self.setup = setup
        self.jvalues = jvalues
        self.jextra = jextra
        
    def __str__(self):
        l_j = self.setup.l_j
        jvtxt = ', '.join(['%s(l=%s)' % (j, l_j[j]) for j in self.jvalues])
        jetxt = ', '.join(['%s(l=%s)' % (j, l_j[j]) for j in self.jextra])
        return '%s: jvalues=[%s], jextra=[%s]' % (self.setup.symbol,
                                                  jvtxt, jetxt)

description = """Generate basis sets that include unoccupied p states as
valence states instead of Gaussian-based polarization functions.
If SYMBOLs are omitted, generate basis sets for all elements with
unoccupied p states and default setup."""


def main():
    parser = OptionParser(usage='%prog [OPTION...] [SYMBOL...]',
                          description=description)
    parser.add_option('--xc', metavar='FUNCTIONAL',
                      default='PBE',
                      help='generate basis sets for FUNCTIONAL[=%default]')
    parser.add_option('--from', metavar='SYMBOL', dest='_from',
                      help='generate starting from SYMBOL if generating '
                      'for all elements')
    opts, symbols = parser.parse_args()

    if len(symbols) == 0:
        symbols = sorted(parameters.keys())
        othersymbols = []
        for symbol in parameters_extra:
            name = parameters_extra[symbol]['name']
            code = '%s.%s' % (symbol, name)
            othersymbols.append(code)
        # Setups that cause trouble
        # trouble = set(['Os.8', 'Ta.5', 'V.5', 'W.6', 'Ir.9'])
        trouble = set([])
        othersymbols = [symbol for symbol in othersymbols
                        if symbol not in trouble]
        symbols.extend(sorted(othersymbols))

        if opts._from:
            index = symbols.index(opts._from)
            symbols = symbols[index:]

    specifications = []
    for sym in symbols:
        try:
            s = SetupData(sym, opts.xc)
        except RuntimeError as e:
            if str(e).startswith('Could not find'):
                #print 'No %s' % sym
                continue
            else:
                raise

        # One could include basis functions also for the ``virtual'' states
        # (marked with negative n)

        jvalues = []
        jextra = []
        for j in range(len(s.f_j)):
            if s.eps_j[j] < 0:
                jvalues.append(j)
                if s.f_j[j] == 0.0 and s.n_j[j] > 0:
                    jextra.append(j)
        if len(jextra) > 0:
            specifications.append(BasisSpecification(s, jvalues, jextra))
            #print sym, jvalues
        # XXX check whether automatic settings coincide with those of official
        # setups distribution
        #bm = BasisMaker(sym, ''

    if world.rank == 0:
        print('Generating basis sets for: %s'
              % ' '.join(spec.setup.symbol for spec in specifications))
    sys.stdout.flush()
    world.barrier()

    for i, spec in enumerate(specifications):
        if i % world.size != world.rank:
            continue
        if world.size > 1:
            print(world.rank, spec)
        else:
            print(spec)
        gtxt = None

        tokens = spec.setup.symbol.split('.')
        sym = tokens[0]

        if len(tokens) == 1:
            p = parameters
            name = 'pvalence'
        elif len(tokens) == 2:
            p = parameters_extra
            name = '%s.pvalence' % tokens[1]
        else:
            raise ValueError('Strange setup specification')

        # This generates only dz setups
        bm = BasisMaker(sym, '%s' % (name),
                        run=False, gtxt=gtxt, xc=opts.xc)
        bm.generator.run(write_xml=False, use_restart_file=False, **p[sym])
        basis = bm.generate(2, 0, txt=None, jvalues=spec.jvalues)
        basis.write_xml()

if __name__ == '__main__':
    main()
