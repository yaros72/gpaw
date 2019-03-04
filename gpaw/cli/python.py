class CLICommand:
    """Run GPAW's Python interpreter."""

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('arguments', nargs='*')

    @staticmethod
    def run(args):
        print('Use: gpaw -P <nproc> python -- <arguments>')
