import cPickle
import numpy as np

d1 = cPickle.load(open('../data/beatles/songs-16s.pickle'))
d0 = cPickle.load(open('../data/beatles/songs-8s.pickle'))

train_prop = 0.9
D = d0
inds = np.random.permutation(len(D))
nd = int(len(D)*train_prop)
tr_inds = inds[:nd]
te_inds = inds[nd::2]
va_inds = inds[nd+1::2]

1/0
