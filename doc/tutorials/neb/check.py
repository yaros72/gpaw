from ase.io import read

images = read('neb.traj@-5:')
ets = images[2].get_potential_energy()
ef = images[4].get_potential_energy()
print(ets - ef)
assert abs(ets - ef - 0.295) < 0.003
