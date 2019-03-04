from gpaw import restart

# J.Phys.: Condens. Matter 18 (2006) 41-54
pw91vasp = {'NO': 0.95,
            'N2': 0.00,
            'O2': 0.00,
            'NRu001': -0.94,
            'ORu001': -2.67,
            'Ru001': 0.00}

pbe = {}
pw91 = {}
for name in ['NO', 'O2', 'N2', 'Ru001', 'NRu001', 'ORu001']:
    a, calc = restart(name + '.gpw', txt=None)
    pbe[name] = a.get_potential_energy()
    pw91[name] = pbe[name] + calc.get_xc_difference('PW91')

energies = {}
for data, text in [(pbe, 'PBE'),
                   (pw91, 'PW91 (non-selfconsitent)'),
                   (pw91vasp, 'PW91 (VASP)')]:
    E = [data['NRu001'] - data['Ru001'] - data['N2'] / 2,
         data['ORu001'] - data['Ru001'] - data['O2'] / 2,
         data['NO'] - data['N2'] / 2 - data['O2'] / 2]
    print('{:24} {:.3f} {:.3f} {:.3f}'.format(text, *E))
    energies[text] = E

for e1, e2 in zip(energies['PW91 (non-selfconsitent)'],
                  energies['PW91 (VASP)']):
    assert abs(e1 - e2) < 0.1
