from __future__ import print_function
import os
import fnmatch
from io import BytesIO
import tarfile
import re
from urllib.request import urlopen


sources = [('gpaw', 'official GPAW setups releases [default]'),
           ('sg15', 'SG15 pseudopotentials'),
           ('basis', 'basis sets for LCAO mode'),
           ('test', 'small file for testing this script')]

names = [r for r, d in sources]

baseurls = {'gpaw':
            'https://wiki.fysik.dtu.dk/gpaw/_sources/setups/setups.rst.txt',
            # 'sg15': 'http://fpmd.ucdavis.edu/qso/potentials/sg15_oncv/',
            'sg15': 'http://www.quantum-simulation.org/potentials/sg15_oncv/',
            'basis':
            'http://dcwww.camd.dtu.dk/~askhl/files/gpaw-lcao-basis-sets/',
            'test': 'http://dcwww.camd.dtu.dk/~askhl/files/gpaw-test-source/'}


notfound_msg = """\
For some reason the files were not found.

Perhaps this script is out of date, and the data is no longer
available at the expected URL:

  {url}

Or maybe there it is just a temporary problem or timeout.  Please try
again, or rummage around the GPAW web page until a solution is found.
Writing e-mails to gpaw-users@listserv.fysik.dtu.dk or reporting
an issue on https://gitlab.com/gpaw/gpaw/issues is also
likely to help."""


class CLICommand:
    """Install PAW datasets, pseudopotential or basis sets.

    Without a directory, show available setups and GPAW
    setup paths.

    With a directory, download and install gpaw-setups into
    INSTALLDIR/[setups-package-name-and-version].
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('directory', nargs='?')
        add('--version',
            help='download VERSION of package.  '
            'Run without arguments to display a list of versions.  '
            'VERSION can be the full URL or a part such as  '
            '\'0.8\' or \'0.6.6300\'')
        add('--tarball', metavar='FILE',
            help='unpack and install from local tarball FILE '
            'instead of downloading')
        add('--list-all', action='store_true',
            help='list packages from all sources')
        for name, help in sources:
            add('--' + name, action='store_const',
                const=name, dest='source',
                help=help)
        add('--register', action='store_true',
            help='run non-interactively and register install path in '
            'GPAW setup search paths.  This is done by adding lines to '
            '~/.gpaw/rc.py')
        add('--no-register', action='store_true',
            help='run non-interactively and do not register install path in '
            'GPAW setup search paths')

    @staticmethod
    def run(args, parser):
        main(args, parser)


def main(args, parser):
    if args.source is None:
        args.source = sources[0][0]

    if args.register and args.no_register:
        parser.error('Conflicting options specified on whether to register '
                     'setup install paths in configuration file.  Try not '
                     'specifying some options.')

    # The sg15 file is a tarbomb.  We will later defuse it by untarring
    # into a subdirectory, so we don't leave a ghastly mess on the
    # unsuspecting user's system.

    if not args.tarball:
        if args.list_all:
            urls = []
            for source in names:
                urls1 = get_urls(source)
                urls.extend(urls1)
        else:
            urls = get_urls(args.source)

        def print_urls(urls, marked=None):
            for url in urls:
                pageurl, fname = url.rsplit('/', 1)
                if url == marked:
                    marking = ' [*]'
                else:
                    marking = '    '
                print(' %s %s' % (marking, url))

        if len(urls) == 0:
            url = baseurls[args.source]
            parser.error(notfound_msg.format(url=url))

        if args.version:
            matching_urls = [url for url in urls if args.version in url]
            if len(matching_urls) > 1:
                parser.error('More than one setup file matches version "%s":\n'
                             '%s' % (args.version, '\n'.join(matching_urls)))
            elif len(matching_urls) == 0:
                parser.error('\nNo setup matched the specified version "%s".\n'
                             'Available setups are:\n'
                             '%s' % (args.version, '\n'.join(urls)))
            assert len(matching_urls) == 1
            url = matching_urls[0]
        else:
            url = urls[0]

        print('Available setups and pseudopotentials')
        print_urls(urls, url)
        print()

    if not args.directory:
        print_setups_info(parser)
        print()
        print('Run gpaw install-data DIR to install newest setups into DIR.')
        print('Run gpaw install-data DIR --version=VERSION to install VERSION '
              '(from above).')
        print('See gpaw install-data --help for more info.')
        raise SystemExit

    targetpath = args.directory

    if args.tarball:
        print('Reading local tarball %s' % args.tarball)
        targzfile = tarfile.open(args.tarball)
        tarfname = args.tarball
    else:
        tarfname = url.rsplit('/', 1)[1]
        print('Selected %s.  Downloading...' % tarfname)
        response = urlopen(url)
        targzfile = tarfile.open(fileobj=BytesIO(response.read()))

    if not os.path.exists(targetpath):
        os.makedirs(targetpath)

    assert tarfname.endswith('.tar.gz')
    setup_dirname = tarfname.rsplit('.', 2)[0]  # remove .tar.gz ending
    setup_path = os.path.abspath(os.path.join(targetpath, setup_dirname))
    if tarfname.startswith('sg15'):
        # Defuse tarbomb
        if not os.path.isdir(setup_path):
            os.mkdir(setup_path)
        targetpath = os.path.join(targetpath, setup_dirname)

    print('Extracting tarball into %s' % targetpath)
    targzfile.extractall(targetpath)
    assert os.path.isdir(setup_path)
    print('Setups installed into %s.' % setup_path)

    # Okay, now we have to maybe edit people's rc files.
    rcfiledir = os.path.join(os.environ['HOME'], '.gpaw')
    rcfilepath = os.path.join(rcfiledir, 'rc.py')

    # We could do all this by importing the rcfile as well and checking
    # whether things are okay or not.
    rcline = "setup_paths.insert(0, '%s')" % setup_path

    # Run interactive mode unless someone specified a flag requiring otherwise
    interactive_mode = not (args.register or args.no_register)

    register_path = False

    if interactive_mode:
        answer = input('Register this setup path in %s? [y/n] ' % rcfilepath)
        if answer.lower() in ['y', 'yes']:
            register_path = True
        elif answer.lower() in ['n', 'no']:
            print('As you wish.')
        else:
            print('What do you mean by "%s"?  Assuming "n".' % answer)
    else:
        if args.register:
            assert not args.no_register
            register_path = True
        else:
            assert args.no_register

    if register_path:
        # First we create the file
        if not os.path.exists(rcfiledir):
            os.makedirs(rcfiledir)
        if not os.path.exists(rcfilepath):
            tmpfd = open(rcfilepath, 'w')  # Just create empty file
            tmpfd.close()

        for line in open(rcfilepath):
            if line.startswith(rcline):
                print('It looks like the path is already registered in %s.'
                      % rcfilepath)
                print('File will not be modified at this time.')
                break
        else:
            rcfd = open(rcfilepath, 'a')
            print(rcline, file=rcfd)
            print('Setup path registered in %s.' % rcfilepath)
            # Need to explicitly flush/close the file so print_setups_info
            # sees the change in rc.py
            rcfd.close()

            print_setups_info(parser)
    else:
        print('You can manually register the setups by adding the')
        print('following line to %s:' % rcfilepath)
        print()
        print(rcline)
        print()
    print('Installation complete.')


def get_urls(source):
    page = baseurls[source]
    response = urlopen(page)
    if source == 'gpaw':
        pattern = 'https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-*.tar.gz'
        lines = (line.strip().decode() for line in response)
        urls = [line for line in lines if fnmatch.fnmatch(line, pattern)]
    elif source == 'sg15':
        # We want sg15_oncv_2015-10-07.tar.gz, but they may upload
        # newer files, too.
        pattern = (r'<a\s*href=[^>]+>\s*'
                   r'(sg15_oncv_upf_\d\d\d\d-\d\d-\d\d.tar.gz)'
                   r'\s*</a>')

        txt = response.read().decode('ascii', errors='replace')
        files = re.compile(pattern).findall(txt)
        files.sort(reverse=True)
        urls = [page + fname for fname in files]
    elif source == 'basis':
        pattern = re.compile('>(gpaw-basis-.+?.tar.gz)</a>')
        txt = response.read().decode('ascii', errors='replace')
        files = sorted(pattern.findall(txt), reverse=True)
        urls = [page + fname for fname in files]
    elif source == 'test':
        urls = ['{0}gpaw-dist-test-source.tar.gz'.format(page)]
    else:
        raise ValueError('Unknown source: %s' % source)
    return urls


def print_setups_info(parser):
    try:
        import gpaw
    except ImportError as e:
        parser.error('Cannot import \'gpaw\'.  GPAW does not appear to be '
                     'installed. %s' % e)

    # GPAW may already have been imported, and the contents of the rc
    # file may have changed since then.  Thus, we re-import gpaw to be
    # sure that everything is as it should be.
    gpaw.initialize_data_paths()
    gpaw.read_rc_file()

    npaths = len(gpaw.setup_paths)
    if npaths == 0:
        print('GPAW currently has no setup search paths')
    else:
        print('Current GPAW setup paths in order of search priority:')
        for i, path in enumerate(gpaw.setup_paths):
            print('%4d. %s' % (i + 1, path))
