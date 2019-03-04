from gpaw.lcao.generate_ngto_augmented import do_nao_ngto_basis

# Generate file with GTO parameters
with open('gbs.txt', 'w') as f:
    def w(s):
        f.write('%s\n' % s)
    w('****')
    w('H     0')
    w('S   1   1.00')
    w('      0.1000000              1.0000000')
    w('P   1   1.00')
    w('      0.0500000              1.0000000')
    w('D   1   1.00')
    w('      0.5000000              1.0000000')
    w('****')

# Run the generator script in order to keep the syntax up-to-date
do_nao_ngto_basis('H', 'LDA', 'sz', 'gbs.txt', 'NAO+NGTO')
