#!/usr/bin/env python
# Emacs: treat this as -*- python -*-

import os
import sys
from optparse import OptionParser, IndentedHelpFormatter
import subprocess


class Formatter(IndentedHelpFormatter):
    """Nicer epilog."""
    def format_epilog(self, epilog):
        return '\n' + '\n'.join(self._format_text(section)
                                for section in epilog) + '\n'


def main():
    description = """Submits a GPAW Python script to the TORQUE queue."""
    
    parser = OptionParser(
        usage='usage: %prog [OPTIONS] <script> [SCRIPT ARGUMENTS]',
        description=description,
        epilog=['Examples:',
                '',
                '    gpaw-qsub -c 32 script.py  # two xeon16',
                '    gpaw-qsub -c 32 -n 2 script.py  # two xeon16, but use '
                'only 1 core on each'],
        formatter=Formatter())
    parser.disable_interspersed_args()

    add = parser.add_option
    add('-c', '--cores', type=int,
        help='Total number of cores to reserve.  Will automatically '
        'determine the architecture (xeon16, xeon8, opteron4).')
    add('-0', '--dry-run', action='store_true',
        help='Do not submit anything, but write parameters and qsub command')
    add('-n', '--processes', type=int,
        help='Number of actual cores to use.')
    add('-q', '--queue',
        help='Name of queue: small, medium, long, verylong '
        '(25m, 2h15m, 13h, 50h).  Default is small')
    add('-l', '--resources',
        help='Specify the resources needed (comma-separated).  Example: '
        '-l nodes=1:ppn=16:xeon16,mem=200gb')
    add('-m', '--mail-options',
        help='Example: -m abe (abort, begin, end)')
    add('-W', '--attributes',
        help='Example: -W depend=afterok:<id>')
    add('-N', '--jobname')
    add('-g', '--gpaw', help='Path to GPAW')
    add('-x', '--export',
        help='Comma separated environment variables to export')
    add('--threads', type=int, default=1,
        help='Number of threads per process.')
    add('--module',
        help='Run library module as a script (terminates option list)')

    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg.startswith('--module='):
            module_args = args[i + 1:]
            del args[i + 1:]
            break
        if arg == '--module':
            module_args = args[i + 2:]
            del args[i + 2:]
            break
    else:
        module_args = []
    
    opts, args = parser.parse_args(args)
    args += module_args
    
    if opts.gpaw:
        path = opts.gpaw
    else:
        import gpaw
        path = gpaw.__path__[0][:-5]
        
    if opts.export:
        export = opts.export.split(',')
    else:
        export = []
        
    if opts.module:
        jobname = opts.module
    elif opts.jobname:
        script = args[0]
        jobname = opts.jobname
    else:
        script = args[0]
        jobname = '_'.join(args)

    qsub = '#!/usr/bin/env python\n'
    qsub += '#PBS -N %s\n' % jobname  # set default job name
    qsub += '#PBS -W umask=002\n'

    if not opts.module:
        if os.path.isfile(script):
            for line in open(os.path.expanduser(script)):
                if line.startswith('#PBS'):
                    qsub += line
        else:
            p = subprocess.Popen(['which', script], stdout=subprocess.PIPE)
            args[0] = p.communicate()[0].strip()

    determine_nodes = True
    if opts.resources:
        resources = opts.resources.split(',')
        for res in resources:
            if res.startswith('nodes='):
                determine_nodes = False
                break
    else:
        resources = []
    
    if opts.cores and determine_nodes:
        for ppn, arch in [(16, 'xeon16'), (8, 'xeon8'), (4, 'opteron4')]:
            if opts.cores % ppn == 0:
                nodes = opts.cores // ppn
                break
        else:
            if opts.cores < 4:
                nodes = 1
                ppn = opts.cores
            else:
                2 / 0
        resources.append('nodes={0}:ppn={1}:{2}'.format(nodes, ppn, arch))

    qsub += 'job = %r\n' % args
    qsub += 'path = %r\n' % path
    qsub += 'module = %r\n' % opts.module
    qsub += 'processes = {0}\n'.format(opts.processes)
    qsub += 'nthreads = %d\n' % opts.threads
    qsub += 'export = %r\n' % export
    header = qsub
    
    qsub += """
import os
import subprocess

nodename = os.uname()[1]
c = nodename[0]
assert c in 'abcdghinmqp'

nproc = len(open(os.environ['PBS_NODEFILE']).readlines())

cmd = ['mpiexec']

export.append('PYTHONPATH=%s:%s' % (path, os.environ.get('PYTHONPATH', '')))

if c in 'ghi':
    # Intel Niflheim node:
    cmd += ['--mca', 'btl', '^tcp']

if processes:
    cmd += ['-np', str(processes), '--loadbalance']
if nthreads > 1:
    export.append('OMP_NUM_THREADS=%d' % nthreads)

for x in export:
    cmd += ['-x', x]
    
cmd.append(os.path.join(path,
                        'build',
                        'bin.linux-x86_64-' + os.environ['FYS_PLATFORM'] + '-2.6',
                        'gpaw-python'))
if module:
    cmd += ['-m', module]
cmd += job

error = subprocess.call(cmd)
if error:
    raise SystemExit(error)
"""
    cmd = ['qsub']
    if opts.queue:
        cmd += ['-q', opts.queue]
    if resources:
        cmd += ['-l', ','.join(resources)]
    if opts.mail_options:
        cmd += ['-m', opts.mail_options]
    if opts.attributes:
        cmd += ['-W', opts.attributes]
    
    if opts.dry_run:
        print(header)
        print(' '.join(cmd))
    else:
        subprocess.Popen(cmd, stdin=subprocess.PIPE).communicate(qsub)


if __name__ == '__main__':
    main()
