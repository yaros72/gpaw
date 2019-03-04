# Creates: ACF.dat
import subprocess
subprocess.call('bader -p all_atom -p atom_index density.cube'.split())
charges = []
for line in open('ACF.dat'):
    words = line.split()
    if len(words) == 7:
        charges.append(float(words[4]))
assert abs(sum(charges) - 10) < 0.0001
assert abs(charges[1] - 0.42) < 0.005
assert abs(charges[2] - 0.42) < 0.005
