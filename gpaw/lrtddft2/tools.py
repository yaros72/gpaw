import sys
import re
import numpy as np
import datetime


################################################################################
def write_parallel_cube(basename, data, gd, atoms, rank):
    hx = gd.h_cv[0,0]
    hy = gd.h_cv[1,1]
    hz = gd.h_cv[2,2]

    f = open('%s_%06d.cube' % (basename, rank), 'w', 256*1024)

    f.write('GPAW Global (natom  nx ny nz  hx hy hz):  %6d  %6d %6d %6d  %12.6lf %12.6lf %12.6lf\n' % (len(atoms), gd.N_c[0], gd.N_c[1], gd.N_c[2], hx,hy,hz))
    f.write('GPAW Local (xbeg xend  ybeg yend  zbeg zend):  %6d %6d  %6d %6d  %6d %6d\n' % ( gd.beg_c[0], gd.end_c[0],  gd.beg_c[1], gd.end_c[1],  gd.beg_c[2], gd.end_c[2]))

    f.write('%6d %12.6lf %12.6lf %12.6lf\n' % (len(atoms), hx*gd.beg_c[0], hy*gd.beg_c[1], hz*gd.beg_c[2]))
    f.write('%6d %12.6lf %12.6lf %12.6lf\n' % (gd.n_c[0], hx, 0.0, 0.0))
    f.write('%6d %12.6lf %12.6lf %12.6lf\n' % (gd.n_c[1], 0.0, hy, 0.0))
    f.write('%6d %12.6lf %12.6lf %12.6lf\n' % (gd.n_c[2], 0.0, 0.0, hz))

    for (i,atom) in enumerate(atoms):
        f.write('%6d %12.6lf %12.6lf %12.6lf %12.6lf\n' % (atom.number, 0.0, atom.position[0]/0.529177, atom.position[1]/0.529177, atom.position[2]/0.529177))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                f.write('%18.8lf\n' % data[i,j,k])
    f.close()


################################################################################
def cubify(out_filename, filenames):
    sys.stderr.write('Reading partial cube files...%s\n' % datetime.datetime.now())
    data = None
    for fname in filenames:
        f = open(fname,'r', 256*1024)
        line0 = f.readline()
        elems = re.compile('[\d.e+-]+').findall(line0)
        natom = int(elems[0])
        (nx,ny,nz) = [int(n) for n in elems[1:4]]
        (hx,hy,hz) = [float(h) for h in elems[4:7]]
        line = f.readline()
        (xb,xe,yb,ye,zb,ze) = [int(x) for x in re.compile('\d+').findall(line)]
        sys.stderr.write('%d %d %d %d %d %d\n' % (xb,xe,yb,ye,zb,ze))

        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()

        atom_lines = []
        for n in range(natom):
            atom_lines.append(f.readline())

        if data is None:
            data = np.zeros((nx,ny,nz))

        for i in range(xb,xe):
            for j in range(yb,ye):
                for k in range(zb,ze):
                    data[i,j,k] = float(f.readline())

        f.close()

    sys.stderr.write('Reading done. %s\n' % datetime.datetime.now())
    sys.stderr.write('Writing the cube file... %d %d %d\n' % (nx,ny,nz))

    out = open(out_filename, 'w', 256*1024)
    out.write(line0)
    out.write('OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')
    out.write('%8d %12.6lf %12.6lf %12.6lf\n' % (natom, 0.0, 0.0, 0.0))
    out.write('%8d %12.6lf %12.6lf %12.6lf\n' % (nx, hx, 0.0, 0.0))
    out.write('%8d %12.6lf %12.6lf %12.6lf\n' % (ny, 0.0, hy, 0.0))
    out.write('%8d %12.6lf %12.6lf %12.6lf\n' % (nz, 0.0, 0.0, hz))
    for n in range(natom):
        out.write(atom_lines[n])
    for i in range(nx):
        sys.stderr.write('.')
        for j in range(ny):
            for k in range(nz):
                out.write('%18.8lf\n' % data[i,j,k])
    out.close()
    sys.stderr.write('\n')
    sys.stderr.write('Writing done. %s\n' % datetime.datetime.now())



################################################################################
def isocubes(filename, isoval, scale):
    """Requires PIL"""
    import Image

    f = open(filename, 'r', 256*1024)
    f.readline()
    f.readline()

    natom = int(f.readline().split()[0])

    elems = f.readline().split()
    nx = int(elems[0])
    # x0 = float(elems[1])

    elems = f.readline().split()
    ny = int(elems[0])
    # y0 = float(elems[2])

    elems = f.readline().split()
    nz = int(elems[0])
    # z0 = float(elems[3])

    for i in range(natom):
        f.readline()


    #data = np.zeros((nx,ny,nz))
    dataxy = [[0.0,0.0] for i in range(nx*ny)]
    dataxz = [[0.0,0.0] for i in range(nx*nz)]
    datayz = [[0.0,0.0] for i in range(ny*nz)]
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                val = float(f.readline())
                if val > isoval:
                    dataxy[i*ny+j][0] += val * scale
                    dataxy[i*ny+j][1] += val * scale
                    dataxz[i*nz+k][0] += val * scale
                    dataxz[i*nz+k][1] += val * scale
                    datayz[j*nz+k][0] += val * scale
                    datayz[j*nz+k][1] += val * scale
                elif -val > isoval:
                    dataxy[i*ny+j][0] += -val * scale
                    dataxy[i*ny+j][1] -= -val * scale
                    dataxz[i*nz+k][0] += -val * scale
                    dataxz[i*nz+k][1] -= -val * scale
                    datayz[j*nz+k][0] += -val * scale
                    datayz[j*nz+k][1] -= -val * scale




    image_xy = Image.new('RGB',(nx,ny))
    image_xy_data = image_xy.load()

    for i in range(nx):
        for j in range(ny):
            n = dataxy[i*ny+j][0] * 100
            d = dataxy[i*ny+j][1] * 100
            if n > 255:
                n = 255
                d = 255*d/n
            n = int(n+.5)
            d = int(d)
            if d > 0:
                dataxy[i*ny+j] = (n,n-d,n-d)
            elif d < 0:
                dataxy[i*ny+j] = (n+d,n+d,n)
            else:
                dataxy[i*ny+j] = (0,0,0)

            image_xy_data[i,j] = dataxy[i*ny+j]

    image_xy.save(filename+'_iso_xy.png')



    image_xz = Image.new('RGB',(nx,nz))
    image_xz_data = image_xz.load()

    for i in range(nx):
        for j in range(nz):
            n = dataxz[i*nz+j][0] * 100
            d = dataxz[i*nz+j][1] * 100
            if n > 255:
                n = 255
                d = 255*d/n
            n = int(n+.5)
            d = int(d)
            if d > 0:
                dataxz[i*nz+j] = (n,n-d,n-d)
            elif d < 0:
                dataxz[i*nz+j] = (n+d,n+d,n)
            else:
                dataxz[i*nz+j] = (0,0,0)

            image_xz_data[i,j] = dataxz[i*nz+j]

    image_xz.save(filename+'_iso_xz.png')


    image_yz = Image.new('RGB',(ny,nz))
    image_yz_data = image_yz.load()

    for i in range(ny):
        for j in range(nz):
            n = datayz[i*nz+j][0] * 100
            d = datayz[i*nz+j][1] * 100
            if n > 255:
                n = 255
                d = 255*d/n
            n = int(n+.5)
            d = int(d)
            if d > 0:
                datayz[i*nz+j] = (n,n-d,n-d)
            elif d < 0:
                datayz[i*nz+j] = (n+d,n+d,n)
            else:
                datayz[i*nz+j] = (0,0,0)

            image_yz_data[i,j] = datayz[i*nz+j]

    image_yz.save(filename+'_iso_yz.png')
