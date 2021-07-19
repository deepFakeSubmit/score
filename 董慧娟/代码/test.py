import pickle

fname = './m.out'
with open(fname, 'rb') as f:
    content = pickle.load(f)
    print(content)