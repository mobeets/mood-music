import cPickle
import argparse
import numpy as np
from keras.layers import Input, Dense, Embedding, Dropout, Conv1D, MaxPooling1D, LSTM, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import to_categorical
from utils.weightnorm import data_based_init
from utils.model_utils import get_callbacks, save_model_in_pieces
from make_songs import song_to_pianoroll, pianoroll_history

def sample_random_rows(D, nsamples):
    """
    keep a random subset of nsamples of each song in D
        note: keeps entire song if len(x) < nsamples
    """
    return [(x[np.random.permutation(len(x))[:nsamples]],y) if len(x) > nsamples else (x,y) for x,y in D]

def get_data_block(D, inds, yrng, batch_size=None):
    X = np.vstack([x for i,(x,y) in enumerate(D) if i in inds])
    I = np.hstack([i*np.ones(len(x)) for i,(x,y) in enumerate(D) if i in inds])
    y = np.hstack([y*np.ones(len(x)) for i,(x,y) in enumerate(D) if i in inds])
    if yrng is not None:
        y = (1.0*y - yrng[0])/(yrng[1] - yrng[0])
    if batch_size is not None:
        nd = batch_size*(len(X)/batch_size)
        X = X[:nd]
        y = y[:nd]
        I = I[:nd]
    return X, y, I

class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]

def load_data(args, train_prop=0.9):
    # load data, make piano rolls
    d = cPickle.load(open(args.train_file))
    D = [(pianoroll_history(song_to_pianoroll(x['song'], trim_flank_silence=True), args.seq_length, args.stride_length), x[args.yname]) for x in d]
    D = [(x,y) for x,y in D if len(x) and y is not None]
    if args.n_per_song != 0:
        min_per_song = np.median([len(x) for x,y in D])
        print "Median parts per song is {}.".format(min_per_song)
        # min_per_song = args.n_per_song
        args.n_per_song = int(min_per_song)
        print "Sampling {} parts per song.".format(args.n_per_song)
        D = sample_random_rows(D, args.n_per_song)

    # jitter song parts +/- up to a major third (4 semitones)
    if args.do_jitter:
        print "Making shifted copies of every part..."
        # offsets = np.random.randint(-4, 5, len(D))
        E = []
        for (x,y) in D:
            E.append((np.roll(x, -4, axis=-1),y))
            E.append((np.roll(x, -2, axis=-1),y))
            E.append((np.roll(x, -1, axis=-1),y))
            E.append((np.roll(x, 1, axis=-1),y))
            E.append((np.roll(x, 2, axis=-1),y))
            E.append((np.roll(x, 4, axis=-1),y))
        D = D + E
    else:
        offsets = np.zeros(len(D))
    # D = [(np.roll(x, offset, axis=-1),y) for (x,y),offset in zip(D, offsets)]

    # reduce dimensionality
    # print "Reducing dimensionality..."
    # dmn = min([np.where(x.sum(axis=0).sum(axis=0))[0].min() for x,y in D])
    # dmx = max([np.where(x.sum(axis=0).sum(axis=0))[0].max() for x,y in D])
    # args.dimrange = (dmn, dmx) # save range
    # print "Reduced dimensionality from {} to {}".format(D[0][0].shape[-1], dmx-dmn+1)
    # D = [(x[:,:,dmn:(dmx+1)],y) for x,y in D]

    if not args.do_classify:
        # normalize y to be in 0..1
        ymn = np.min([y for x,y in D])
        ymx = np.max([y for x,y in D])
        yrng = (ymn, ymx)
    else:
        yrng = None

    # split by song into train/validation/test
    print "Splitting into train and test..."
    inds = np.random.permutation(len(D))
    nd = int(len(D)*train_prop)
    tr_inds = inds[:nd]
    # va_inds = inds[nd:]
    te_inds = inds[nd::2]
    va_inds = inds[nd+1::2]
    Xtr, ytr, Itr = get_data_block(D, tr_inds, yrng,
        args.batch_size)
    Xva, yva, Iva = get_data_block(D, va_inds, yrng,
        args.batch_size)
    Xte, yte, Ite = get_data_block(D, te_inds, yrng,
        args.batch_size)
    # Xte, yte, Ite = [], [], []
    return AttrDict({'Xtr': Xtr, 'ytr': ytr,
        'Xva': Xva, 'yva': yva, 'Xte': Xte, 'yte': yte,
        'Itr': Itr, 'Ite': Ite, 'Iva': Iva})

def get_model(args):
    n_filters = 16
    kernel_size = 2
    pool_size = 2
    hidden_dim = 8

    X = Input(batch_shape=(args.batch_size,
        args.seq_length, args.original_dim), name='X')
    # X.shape == [100, S, 128]

    P1 = Dense(args.original_dim)(X)
    P1b = BatchNormalization()(P1)
    P1a = Activation('relu')(P1b)
    P1d = Dropout(args.dropout)(P1a)
    zf = Flatten()(P1d)

    # z = Dense(hidden_dim)(P1d)
    # zb = BatchNormalization()(z)
    # za = Activation('relu')(zb)
    # zd = Dropout(args.dropout)(za)
    # zf = Flatten()(zd)

    if args.do_classify:
        y = Dense(args.n_classes,
            activation='softmax', name='y')(zf)
    else:
        y = Dense(1, activation='sigmoid', name='y')(zf)
    model = Model(X, y)
    if args.do_classify:
        model.compile(optimizer=args.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    else:
        model.compile(optimizer=args.optimizer,
            loss='binary_crossentropy')
    return model

    C = Conv1D(n_filters, kernel_size,
        input_shape=(args.seq_length, args.original_dim),
        padding='valid', activation='relu', strides=1)(X)
    # C.shape == [100, ~S, n_filters]

    P = MaxPooling1D(pool_size=pool_size)(C)
    # P.shape == [100, ~S/pool_size, n_filters]

    C2 = Conv1D(n_filters/2, kernel_size,
        padding='valid', activation='relu', strides=1)(P)
    P2 = MaxPooling1D(pool_size=pool_size)(C2)
    z = Dense(hidden_dim, activation='relu')(P2)
    zf = Flatten()(z)
    if args.do_classify:
        y = Dense(args.n_classes,
                activation='softmax', name='y')(zf)
    else:
        y = Dense(1, activation='sigmoid', name='y')(zf)

    model = Model(X, y)
    if args.do_classify:
        model.compile(optimizer=args.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    else:
        model.compile(optimizer=args.optimizer,
            loss='binary_crossentropy')
    return model

def plot(D, model):
    import matplotlib.pyplot as plt
    yha = model.predict(D.Xva, batch_size=args.batch_size)
    plt.plot(D.yva, yha, '.', color='0.8')
    for ind in np.unique(D.Iva):
        plt.plot(D.yva[D.Iva == ind].mean(), yha[D.Iva == ind].mean(), 'ko')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.savefig('../data/plots/yva_hat.png')
    yha2 = model.predict(D.Xtr, batch_size=args.batch_size)
    plt.clf()
    plt.plot(D.ytr, yha2, '.', color='0.8')
    for ind in np.unique(D.Itr):
        plt.plot(D.ytr[D.Itr == ind].mean(), yha2[D.Itr == ind].mean(), 'ko')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.savefig('../data/plots/ytr_hat.png')

def train(args):
    """
    - get convolutions working over space
    - then augment data by adding +/- a major third in key
    - add batch normalization: https://keras.io/layers/normalization/

    - prevent overfitting...maybe I just need more songs and not just shifted copies of things already in training data.
    - try on a bigger midi song data set
    """
    D = load_data(args)
    if args.do_classify:
        ys = [D.ytr, D.yte, D.yva]
        args.n_classes = len(np.unique(np.hstack(ys)))
        D.ytr = to_categorical(D.ytr, args.n_classes)
        D.yte = to_categorical(D.yte, args.n_classes)
        D.yva = to_categorical(D.yva, args.n_classes)

    print "Training X ({}) and y ({})".format(D.Xtr.shape, D.ytr.shape)

    args.original_dim = D.Xtr.shape[-1]
    model = get_model(args)
    model_file = save_model_in_pieces(model, args)
    data_based_init(model, D.Xtr[:100])
    history = model.fit(D.Xtr, D.ytr,
        shuffle=True,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        callbacks=get_callbacks(args),
        validation_data=(D.Xva, D.yva))
    model.load_weights(model_file)
    if not args.do_classify:
        plot(D, model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
        help='tag for current run')
    parser.add_argument('--num_epochs', type=int, default=50,
        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=100,
        help='batch size')
    parser.add_argument('--seq_length', type=int, default=16,
        help='seq length')
    parser.add_argument('--stride_length', type=int, default=1,
        help='stride length')
    parser.add_argument('--n_per_song', type=int, default=1,
        help='samples per song')
    parser.add_argument('--dropout', type=float, default=0.0,
        help='dropout (proportion)')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--yname', type=str,
        choices=['playcount', 'mood_valence', 'mood_energy'],
        default='playcount', help='what to fit')
    parser.add_argument("--do_classify", action="store_true",
                help="treat y as category")
    parser.add_argument("--do_jitter", action="store_true",
                help="add jitter to piano rolls")
    parser.add_argument('--optimizer', type=str,
        default='adam', help='optimizer name')
    parser.add_argument('--train_file', type=str,
        default='../data/beatles/songs-16s.pickle',
        help='file of training data (.pickle)')
    parser.add_argument('--model_dir', type=str,
        default='../data/models')
    args = parser.parse_args()
    train(args)
