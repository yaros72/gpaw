import pickle

results = pickle.load(open('C-g0w0_results.pckl', 'rb'))
direct_gap = results['qp'][0, 0, -1] - results['qp'][0, 0, -2]

print('Direct bandgap of C:', direct_gap)
