from gpaw import GPAW

energies = []
for name in ['zero', 'periodic', 'corrected']:
    calc = GPAW(name + '.gpw', txt=None)
    energies.append(calc.get_potential_energy())

print(energies)
assert abs(energies[1] - energies[0]) < 0.003
assert abs(energies[2] - energies[0] - 0.0409) < 0.003

efermi = calc.get_fermi_level()

v = calc.get_electrostatic_potential().mean(0).mean(0)
w1 = v[0] - efermi
w2 = v[-1] - efermi
print(w1, w2)
assert abs(w1 - 4.359) < 0.01
assert abs(w2 - 2.556) < 0.01
