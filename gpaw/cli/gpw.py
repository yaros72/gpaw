import os


class CLICommand:
    """Manipulate/show content of GPAW-restart file."""

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('gpw', metavar='gpw-file')
        parser.add_argument('-w', '--remove-wave-functions',
                            action='store_true')

    @staticmethod
    def run(args):
        if args.remove_wave_functions:
            import ase.io.ulm as ulm
            reader = ulm.open(args.gpw)
            if 'values' not in reader.wave_functions:
                print('No wave functions in', args.gpw)
            else:
                ulm.copy(reader, args.gpw + '.temp',
                         exclude={'.wave_functions.values'})
                reader.close()
                os.rename(args.gpw + '.temp', args.gpw)
        else:
            from gpaw import GPAW
            GPAW(args.gpw)
