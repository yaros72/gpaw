def create_wave_function_mode(name, **kwargs):
    if name not in ['fd', 'pw', 'lcao']:
        raise ValueError('Unknown wave function mode: ' + name)

    from gpaw.wavefunctions.fd import FD
    from gpaw.wavefunctions.pw import PW
    from gpaw.wavefunctions.lcao import LCAO
    return {'fd': FD, 'pw': PW, 'lcao': LCAO}[name](**kwargs)


class Mode:
    def __init__(self, force_complex_dtype=False):
        self.force_complex_dtype = force_complex_dtype

    def todict(self):
        dct = {'name': self.name}
        if self.force_complex_dtype:
            dct['force_complex_dtype'] = True
        return dct
