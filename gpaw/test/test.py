from __future__ import print_function
import os
import tempfile
import warnings

import gpaw.mpi as mpi
from gpaw.cli.info import info
from gpaw import debug


class CLICommand:
    """Run the GPAW test suite.

    The test suite can be run in
    parallel with MPI through gpaw-python.  The test suite
    supports 1, 2, 4 or 8 CPUs although some tests are
    skipped for some parallelizations.  If no TESTs are
    given, run all tests supporting the parallelization.
    """

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('tests', nargs='*')
        add('-x', '--exclude',
            type=str, default=None,
            help='Exclude tests (comma separated list of tests).',
            metavar='test1.py,test2.py,...')
        add('-f', '--run-failed-tests-only',
            action='store_true',
            help='Run failed tests only.')
        add('--from', metavar='TESTFILE', dest='from_test',
            help='Run remaining tests, starting from TESTFILE')
        add('--after', metavar='TESTFILE', dest='after_test',
            help='Run remaining tests, starting after TESTFILE')
        add('--range',
            type=str, default=None,
            help='Run tests in range test_i.py to test_j.py '
            '(inclusive)',
            metavar='test_i.py,test_j.py')
        add('-j', '--jobs', type=int, default=1,
            help='Run JOBS threads.  Each test will be executed '
            'in serial by one thread.  This option cannot be used '
            'for parallelization together with MPI.')
        add('--reverse', action='store_true',
            help='Run tests in reverse order (less overhead with '
            'multiple jobs)')
        add('-k', '--keep-temp-dir', action='store_true',
            dest='keep_tmpdir',
            help='Do not delete temporary files.')
        add('-d', '--directory', help='Run test in this directory')
        add('-s', '--show-output', action='store_true',
            help='Show standard output from tests.')
        add('--list', action='store_true',
            help='list the full list of tests, then exit')

    @staticmethod
    def run(args):
        main(args)


def main(args):
    if len(args.tests) == 0:
        from gpaw.test import tests
    else:
        tests = args.tests

    if args.list:
        mydir, _ = os.path.split(__file__)
        for test in tests:
            print(os.path.join(mydir, test))
        return

    if args.reverse:
        tests.reverse()

    if args.run_failed_tests_only:
        tests = [line.strip() for line in open('failed-tests.txt')]

    exclude = []
    if args.exclude is not None:
        exclude += args.exclude.split(',')

    if args.from_test:
        fromindex = tests.index(args.from_test)
        tests = tests[fromindex:]

    if args.after_test:
        index = tests.index(args.after_test) + 1
        tests = tests[index:]

    if args.range:
        # default start(stop) index is first(last) test
        indices = args.range.split(',')
        try:
            start_index = tests.index(indices[0])
        except ValueError:
            start_index = 0
        try:
            stop_index = tests.index(indices[1]) + 1
        except ValueError:
            stop_index = len(tests)
        tests = tests[start_index:stop_index]

    if args.jobs > 1:
        exclude.append('maxrss.py')

    for test in exclude:
        if test in tests:
            tests.remove(test)

    from gpaw.test import TestRunner

    if mpi.world.size > 8:
        if mpi.rank == 0:
            message = (
                '!!!!!!!\n'
                'GPAW regression test suite was not designed to run on more\n'
                'than 8 MPI tasks. Re-run test suite using 1, 2, 4 or 8 MPI\n'
                'tasks instead.')
            warnings.warn(message, RuntimeWarning)

    if mpi.rank == 0:
        if args.directory is None:
            tmpdir = tempfile.mkdtemp(prefix='gpaw-test-')
        else:
            tmpdir = args.directory
            if os.path.isdir(tmpdir):
                args.keep_tmpdir = True
            else:
                os.mkdir(tmpdir)
    else:
        tmpdir = None
    tmpdir = mpi.broadcast_string(tmpdir)
    cwd = os.getcwd()
    os.chdir(tmpdir)
    if mpi.rank == 0:
        info()
        print('Running tests in', tmpdir)
        print('Jobs: {}, Cores: {}, debug-mode: {}'
              .format(args.jobs, mpi.size, debug))
    failed = TestRunner(tests, jobs=args.jobs,
                        show_output=args.show_output).run()
    os.chdir(cwd)
    mpi.world.barrier()  # syncronize before removing tmpdir
    if mpi.rank == 0:
        if len(failed) > 0:
            open('failed-tests.txt', 'w').write('\n'.join(failed) + '\n')
        if not args.keep_tmpdir:
            os.system('rm -rf ' + tmpdir)
    return failed
