from myqueue.task import task


def create_tasks():
    return [
        task('Pt_gs.py@4:20m'),
        task('Pt_bands.py@32:1h', deps='Pt_gs.py'),
        task('plot_Pt_bands.py@1:10m', deps='Pt_bands.py'),
        task('WS2_gs.py@4:20h'),
        task('WS2_bands.py@32:3h', deps='WS2_gs.py'),
        task('plot_WS2_bands.py@1:10m', deps='WS2_bands.py'),
        task('Fe_gs.py@4:20m'),
        task('Fe_bands.py@32:1h', deps='Fe_gs.py'),
        task('plot_Fe_bands.py@1:10m', deps='Fe_bands.py'),
        task('gs_Bi2Se3.py@4:40m'),
        task('Bi2Se3_bands.py@32:5h', deps='gs_Bi2Se3.py'),
        task('high_sym.py@4:30h', deps='gs_Bi2Se3.py'),
        task('parity.py@1:5h', deps='high_sym.py'),
        task('plot_Bi2Se3_bands.py@1:2h', deps='Bi2Se3_bands.py'),
        task('gs_Co.py@32:2h'),
        task('anisotropy.py@1:5h', deps='gs_Co.py'),
        task('plot_anisotropy.py@1:2m', deps='anisotropy.py')]
