# Copyright (C) 2006 CSC-Scientific Computing Ltd.
# Please see the accompanying LICENSE file for further information.
from __future__ import print_function
import os
import platform
import sys
import re
import distutils.util
from distutils.sysconfig import get_config_vars
from distutils.version import LooseVersion
from glob import glob
from os.path import join
from stat import ST_MTIME


def find_file(arg, dir, files):
    # looks if the first element of the list arg is contained in the list files
    # and if so, appends dir to to arg. To be used with the os.path.walk
    if arg[0] in files:
        arg.append(dir)


def get_system_config(define_macros, undef_macros,
                      include_dirs, libraries, library_dirs, extra_link_args,
                      extra_compile_args, runtime_library_dirs, extra_objects,
                      import_numpy):

    undef_macros += ['NDEBUG']
    if import_numpy:
        import numpy
        include_dirs += [numpy.get_include()]

    # libxc
    libraries += ['xc']

    machine = platform.uname()[4]
    if machine == 'sun4u':

        #  _
        # |_ | ||\ |
        #  _||_|| \|
        #

        extra_compile_args += ['-Kpic', '-fast']

        # Suppress warning from -fast (-xarch=native):
        f = open('cc-test.c', 'w')
        f.write('int main(){}\n')
        f.close()
        stderr = os.popen3('cc cc-test.c -fast')[2].read()
        arch = re.findall('-xarch=(\S+)', stderr)
        os.remove('cc-test.c')
        if len(arch) > 0:
            extra_compile_args += ['-xarch=%s' % arch[-1]]

        # We need the -Bstatic before the -lsunperf and -lfsu:
        # http://forum.java.sun.com/thread.jspa?threadID=5072537&messageID=9265782
        extra_link_args += ['-Bstatic', '-lsunperf', '-lfsu', '-Bdynamic']
        cc_version = os.popen3('cc -V')[2].readline().split()[3]
        if LooseVersion(cc_version) > '5.6':
            libraries.append('mtsk')
        else:
            extra_link_args.append('-lmtsk')
        # define_macros.append(('NO_C99_COMPLEX', '1'))

    elif sys.platform.startswith('win'):

        # We compile with mingw coming from pythonyx (32-bit)
        # on the msys command line, e.g.:
        # LIBRARY_PATH=/c/libxc/lib:/c/OpenBLAS/lib \
        # C_INCLUDE_PATH=/c/libxc/include python setup.py build
        if 'LIBRARY_PATH' in os.environ:
            library_dirs += os.environ['LIBRARY_PATH'].split(os.path.pathsep)

        extra_compile_args += ['-Wall', '-std=c99']

        lib = ''
        for ld in library_dirs:
            # OpenBLAS (includes Lapack)
            if os.path.exists(join(ld, 'libopenblas.a')):
                lib = 'openblas'
                break
        if lib == 'openblas':
            libraries += [lib, 'gfortran']

    elif sys.platform in ['aix5', 'aix6']:

        #
        # o|_  _ _
        # ||_)| | |
        #

        extra_compile_args += ['-qlanglvl=stdc99']
        # setting memory limit is necessary on aix5
        if sys.platform == 'aix5':
            extra_link_args += ['-bmaxdata:0x80000000',
                                '-bmaxstack:0x80000000']

        libraries += ['f', 'lapack', 'essl']
        define_macros.append(('GPAW_AIX', '1'))

    elif sys.platform == 'darwin':

        extra_compile_args += ['-Wall', '-Wno-unknown-pragmas', '-std=c99']
        include_dirs += ['/usr/include/malloc']
        library_dirs += ['/usr/local/lib']   # libxc installed with Homebrew
        extra_link_args += ['-framework', 'Accelerate']  # BLAS

        # If the user uses python 3 from homebrew, then installation would
        # fail due to a broken LINKFORSHARED configuration variable
        cfgDict = get_config_vars()
        if ('LINKFORSHARED' in cfgDict and
            'Python.framework/Versions/3.6' in cfgDict['LINKFORSHARED']):
            cfgDict['LINKFORSHARED'] = '-l python3.6'

        # We should probably add something to allow for user-installed BLAS?

    elif machine in ['x86_64', 'ppc64', 'ppc64le', 'aarch64', 's390x']:

        #    _
        # \/|_||_    |_ |_|
        # /\|_||_| _ |_|  |
        #

        extra_compile_args += ['-Wall', '-Wno-unknown-pragmas', '-std=c99']

        # Look for ACML libraries:
        acml = glob('/opt/acml*/g*64/lib')
        if len(acml) > 0:
            library_dirs += [acml[-1]]
            libraries += ['acml']
            if acml[-1].find('gfortran') != -1:
                libraries.append('gfortran')
            if acml[-1].find('gnu') != -1:
                libraries.append('g2c')
            extra_link_args += ['-Wl,-rpath=' + acml[-1]]
        else:
            atlas = False
            for dir in ['/usr/lib', '/usr/local/lib', '/usr/lib64/atlas']:
                if glob(join(dir, 'libatlas.so')) != []:
                    atlas = True
                    libdir = dir
                    break
            satlas = False
            for dir in ['/usr/lib', '/usr/local/lib', '/usr/lib64/atlas']:
                if glob(join(dir, 'libsatlas.so')) != []:
                    satlas = True
                    libdir = dir
                    break
            openblas = False
            for dir in ['/usr/lib', '/usr/local/lib', '/usr/lib64']:
                if glob(join(dir, 'libopenblas.so')) != []:
                    openblas = True
                    libdir = dir
                    break
            if 'MKLROOT' in os.environ:
                libraries += ['mkl_intel_lp64', 'mkl_sequential', 'mkl_core',
                              'irc']
            elif openblas:  # prefer openblas
                libraries += ['openblas', 'lapack']
                library_dirs += [libdir]
            else:
                if atlas:  # then atlas
                    # http://math-atlas.sourceforge.net/errata.html#LINK
                    # atlas does not respect OMP_NUM_THREADS - build
                    # single-thread
                    # http://math-atlas.sourceforge.net/faq.html#tsafe
                    libraries += ['lapack', 'f77blas', 'cblas', 'atlas']
                    library_dirs += [libdir]
                elif satlas:  # then atlas >= 3.10 Fedora/RHEL
                    libraries += ['satlas']
                    library_dirs += [libdir]
                else:
                    libraries += ['blas', 'lapack']

    elif machine == 'ia64':

        #  _  _
        # |_ |  o
        #  _||_||
        #

        extra_compile_args += ['-Wall', '-std=c99']
        libraries += ['mkl', 'mkl_lapack64']

    elif platform.machine().startswith('arm'):

        extra_compile_args += ['-Wall', '-Wno-unknown-pragmas', '-std=c99']

        atlas = False
        for dir in ['/usr/lib', '/usr/local/lib', '/usr/lib/atlas']:
            if glob(join(dir, 'libatlas.so')) != []:
                atlas = True
                libdir = dir
                break
        satlas = False
        for dir in ['/usr/lib', '/usr/local/lib', '/usr/lib/atlas']:
            if glob(join(dir, 'libsatlas.so')) != []:
                satlas = True
                libdir = dir
                break
        openblas = False
        for dir in ['/usr/lib', '/usr/local/lib']:
            if glob(join(dir, 'libopenblas.so')) != []:
                openblas = True
                libdir = dir
                break
        if openblas:  # prefer openblas
            libraries += ['openblas']
            library_dirs += [libdir]
        else:
            if atlas:  # then atlas
                # http://math-atlas.sourceforge.net/errata.html#LINK
                # atlas does not respect OMP_NUM_THREADS - build single-thread
                # http://math-atlas.sourceforge.net/faq.html#tsafe
                libraries += ['lapack', 'f77blas', 'cblas', 'atlas']
                library_dirs += [libdir]
            elif satlas:  # then atlas >= 3.10 Fedora/RHEL
                libraries += ['satlas']
                library_dirs += [libdir]
            else:
                libraries += ['blas', 'lapack']

    elif machine == 'i686':

        #      _
        # o|_ |_||_
        # ||_||_||_|
        #

        extra_compile_args += ['-Wall', '-Wno-unknown-pragmas', '-std=c99']

        if 'MKL_ROOT' in os.environ:
            mklbasedir = [os.environ['MKL_ROOT']]
        else:
            mklbasedir = glob('/opt/intel/mkl*')
        libs = ['libmkl_ia32.a']
        if mklbasedir != []:
            os.path.walk(mklbasedir[0], find_file, libs)
        libs.pop(0)
        if libs != []:
            libs.sort()
            libraries += ['mkl_lapack',
                          'mkl_ia32', 'guide', 'pthread', 'mkl']
            library_dirs += libs
            # extra_link_args += ['-Wl,-rpath=' + library_dirs[-1]]
        else:
            atlas = False
            for dir in ['/usr/lib', '/usr/local/lib', '/usr/lib/atlas']:
                if glob(join(dir, 'libatlas.so')) != []:
                    atlas = True
                    libdir = dir
                    break
            satlas = False
            for dir in ['/usr/lib', '/usr/local/lib', '/usr/lib/atlas']:
                if glob(join(dir, 'libsatlas.so')) != []:
                    satlas = True
                    libdir = dir
                    break
            openblas = False
            for dir in ['/usr/lib', '/usr/local/lib']:
                if glob(join(dir, 'libopenblas.so')) != []:
                    openblas = True
                    libdir = dir
                    break
            if openblas:  # prefer openblas
                libraries += ['openblas']
                library_dirs += [libdir]
            else:
                if atlas:  # then atlas
                    # http://math-atlas.sourceforge.net/errata.html#LINK
                    # atlas does not respect OMP_NUM_THREADS - build
                    # single-thread
                    # http://math-atlas.sourceforge.net/faq.html#tsafe
                    libraries += ['lapack', 'f77blas', 'cblas', 'atlas']
                    library_dirs += [libdir]
                elif satlas:  # then atlas >= 3.10 Fedora/RHEL
                    libraries += ['satlas']
                    library_dirs += [libdir]
                else:
                    libraries += ['blas', 'lapack']

            # add libg2c if available
            g2c = False
            for dir in ['/usr/lib', '/usr/local/lib']:
                if glob(join(dir, 'libg2c.so')) != []:
                    g2c = True
                    break
                if glob(join(dir, 'libg2c.a')) != []:
                    g2c = True
                    break
            if g2c:
                libraries += ['g2c']

    else:
        extra_compile_args += ['-Wall', '-Wno-unknown-pragmas', '-std=c99']

        atlas = False
        for dir in ['/usr/lib', '/usr/local/lib', '/usr/lib/atlas']:
            if glob(join(dir, 'libatlas.so')) != []:
                atlas = True
                libdir = dir
                break
        satlas = False
        for dir in ['/usr/lib', '/usr/local/lib', '/usr/lib/atlas']:
            if glob(join(dir, 'libsatlas.so')) != []:
                satlas = True
                libdir = dir
                break
        openblas = False
        for dir in ['/usr/lib', '/usr/local/lib']:
            if glob(join(dir, 'libopenblas.so')) != []:
                openblas = True
                libdir = dir
                break
        if openblas:  # prefer openblas
            libraries += ['openblas']
            library_dirs += [libdir]
        else:
            if atlas:  # then atlas
                # http://math-atlas.sourceforge.net/errata.html#LINK
                # atlas does not respect OMP_NUM_THREADS - build single-thread
                # http://math-atlas.sourceforge.net/faq.html#tsafe
                libraries += ['lapack', 'f77blas', 'cblas', 'atlas']
                library_dirs += [libdir]
            elif satlas:  # then atlas >= 3.10 Fedora/RHEL
                libraries += ['satlas']
                library_dirs += [libdir]
            else:
                libraries += ['blas', 'lapack']

    # https://listserv.fysik.dtu.dk/pipermail/gpaw-users/2012-May/001473.html
    p = platform.dist()
    if p[0].lower() in ['redhat', 'centos'] and p[1].startswith('6.'):
        define_macros.append(('_GNU_SOURCE', '1'))


def mtime(path, name, mtimes):
    """Return modification time.

    The modification time of a source file is returned.  If one of its
    dependencies is newer, the mtime of that file is returned.
    This function fails if two include files with the same name
    are present in different directories."""

    include = re.compile('^#\s*include "(\S+)"', re.MULTILINE)

    if name in mtimes:
        return mtimes[name]
    t = os.stat(os.path.join(path, name))[ST_MTIME]
    for name2 in include.findall(open(os.path.join(path, name)).read()):
        path2, name22 = os.path.split(name2)
        if name22 != name:
            t = max(t, mtime(os.path.join(path, path2), name22, mtimes))
    mtimes[name] = t
    return t


def check_dependencies(sources):
    # Distutils does not do deep dependencies correctly.  We take care of
    # that here so that "python setup.py build_ext" always does the right
    # thing!
    mtimes = {}  # modification times

    # Remove object files if any dependencies have changed:
    plat = distutils.util.get_platform() + '-' + sys.version[0:3]
    remove = False
    for source in sources:
        path, name = os.path.split(source)
        t = mtime(path + '/', name, mtimes)
        o = 'build/temp.%s/%s.o' % (plat, source[:-2])  # object file
        if os.path.exists(o) and t > os.stat(o)[ST_MTIME]:
            print('removing', o)
            os.remove(o)
            remove = True

    so = 'build/lib.%s/_gpaw.so' % plat
    if os.path.exists(so) and remove:
        # Remove shared object C-extension:
        # print 'removing', so
        os.remove(so)


def test_configuration():
    raise NotImplementedError


def write_configuration(define_macros, include_dirs, libraries, library_dirs,
                        extra_link_args, extra_compile_args,
                        runtime_library_dirs, extra_objects, mpicompiler,
                        mpi_libraries, mpi_library_dirs, mpi_include_dirs,
                        mpi_runtime_library_dirs, mpi_define_macros):

    # Write the compilation configuration into a file
    try:
        out = open('configuration.log', 'w')
    except IOError as x:
        print(x)
        return
    print("Current configuration", file=out)
    print("libraries", libraries, file=out)
    print("library_dirs", library_dirs, file=out)
    print("include_dirs", include_dirs, file=out)
    print("define_macros", define_macros, file=out)
    print("extra_link_args", extra_link_args, file=out)
    print("extra_compile_args", extra_compile_args, file=out)
    print("runtime_library_dirs", runtime_library_dirs, file=out)
    print("extra_objects", extra_objects, file=out)
    if mpicompiler is not None:
        print(file=out)
        print("Parallel configuration", file=out)
        print("mpicompiler", mpicompiler, file=out)
        print("mpi_libraries", mpi_libraries, file=out)
        print("mpi_library_dirs", mpi_library_dirs, file=out)
        print("mpi_include_dirs", mpi_include_dirs, file=out)
        print("mpi_define_macros", mpi_define_macros, file=out)
        print("mpi_runtime_library_dirs", mpi_runtime_library_dirs, file=out)
    out.close()


def build_interpreter(define_macros, include_dirs, libraries, library_dirs,
                      extra_link_args, extra_compile_args,
                      runtime_library_dirs, extra_objects,
                      mpicompiler, mpilinker, mpi_libraries, mpi_library_dirs,
                      mpi_include_dirs, mpi_runtime_library_dirs,
                      mpi_define_macros):

    # Build custom interpreter which is used for parallel calculations

    cfgDict = get_config_vars()
    plat = distutils.util.get_platform() + '-' + sys.version[0:3]

    cfiles = glob('c/[a-zA-Z_]*.c') + ['c/bmgs/bmgs.c']
    cfiles += glob('c/xc/*.c')
    # Make build process deterministic (for "reproducible build" in debian)
    # XXX some of this is duplicated in setup.py!  Why do the same thing twice?
    cfiles.sort()

    sources = ['c/bc.c', 'c/localized_functions.c', 'c/mpi.c', 'c/_gpaw.c',
               'c/operators.c', 'c/woperators.c', 'c/transformers.c',
               'c/elpa.c',
               'c/blacs.c', 'c/utilities.c', 'c/xc/libvdwxc.c']
    objects = ' '.join(['build/temp.%s/' % plat + x[:-1] + 'o'
                        for x in cfiles])

    if not os.path.isdir('build/bin.%s/' % plat):
        os.makedirs('build/bin.%s/' % plat)
    exefile = 'build/bin.%s/' % plat + '/gpaw-python'

    libraries += mpi_libraries
    library_dirs += mpi_library_dirs
    define_macros += mpi_define_macros
    include_dirs += mpi_include_dirs
    runtime_library_dirs += mpi_runtime_library_dirs

    define_macros.append(('PARALLEL', '1'))
    define_macros.append(('GPAW_INTERPRETER', '1'))
    macros = ' '.join(['-D%s=%s' % x for x in define_macros if x[0].strip()])

    include_dirs.append(cfgDict['INCLUDEPY'])
    include_dirs.append(cfgDict['CONFINCLUDEPY'])
    includes = ' '.join(['-I' + incdir for incdir in include_dirs])

    library_dirs.append(cfgDict['LIBPL'])
    lib_dirs = ' '.join(['-L' + lib for lib in library_dirs])

    libs = ' '.join(['-l' + lib for lib in libraries if lib.strip()])
    # See if there is "scalable" libpython available
    libpl = cfgDict['LIBPL']
    if glob(libpl + '/libpython*mpi*'):
        libs += ' -lpython%s_mpi' % cfgDict['VERSION']
    else:
        libs += ' ' + cfgDict.get('BLDLIBRARY',
                                  '-lpython%s' % cfgDict['VERSION'])
    libs = ' '.join([libs, cfgDict['LIBS'], cfgDict['LIBM']])

    # Hack taken from distutils to determine option for runtime_libary_dirs
    if sys.platform[:6] == 'darwin':
        # MacOSX's linker doesn't understand the -R flag at all
        runtime_lib_option = '-L'
    elif sys.platform[:5] == 'hp-ux':
        runtime_lib_option = '+s -L'
    elif os.popen('mpicc --showme 2> /dev/null', 'r').read()[:3] == 'gcc':
        runtime_lib_option = '-Wl,-R'
    elif os.popen('mpicc -show 2> /dev/null', 'r').read()[:3] == 'gcc':
        runtime_lib_option = '-Wl,-R'
    else:
        runtime_lib_option = '-R'

    runtime_libs = ' '.join([runtime_lib_option + lib
                             for lib in runtime_library_dirs])

    extra_link_args.append(cfgDict['LDFLAGS'])

    if sys.platform in ['aix5', 'aix6']:
        extra_link_args.append(cfgDict['LINKFORSHARED'].replace(
            'Modules', cfgDict['LIBPL']))
    elif sys.platform == 'darwin':
        # On a Mac, it is important to preserve the original compile args.
        # This should probably always be done ?!?
        extra_compile_args.append(cfgDict['CFLAGS'])
        extra_link_args.append(cfgDict['LINKFORSHARED'])
    else:
        extra_link_args.append(cfgDict['LINKFORSHARED'])

    extra_compile_args.append('-fPIC')

    # Compile the parallel sources
    for src in sources:
        obj = 'build/temp.%s/' % plat + src[:-1] + 'o'
        cmd = ('%s %s %s %s -o %s -c %s ') % \
              (mpicompiler,
               macros,
               ' '.join(extra_compile_args),
               includes,
               obj,
               src)
        print(cmd)
        error = os.system(cmd)
        if error != 0:
            return error

    # Link the custom interpreter
    cmd = ('%s -o %s %s %s %s %s %s %s') % \
          (mpilinker,
           exefile,
           objects,
           ' '.join(extra_objects),
           lib_dirs,
           libs,
           runtime_libs,
           ' '.join(extra_link_args))

    print(cmd)
    error = os.system(cmd)
    return error
