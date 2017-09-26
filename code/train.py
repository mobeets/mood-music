import cPickle
import argparse
import numpy as np
from keras.layers import Input, Dense, Embedding, Dropout, Conv1D, MaxPooling1D, LSTM, Flatten
from keras.models import Model
from utils.model_utils import get_callbacks, save_model_in_pieces
from make_songs import song_to_pianoroll, pianoroll_history

def sample_random_rows(D, nsamples):
    return [(x[np.random.permutation(len(x))[:nsamples]],y) for x,y in D if len(x) > nsamples]

def get_data_block(D, inds, yrng, batch_size=None):
    X = np.vstack([x for i,(x,y) in enumerate(D) if i in inds])
    y = np.hstack([y*np.ones(len(x)) for i,(x,y) in enumerate(D) if i in inds])
    y = (1.0*y - yrng[0])/(yrng[1] - yrng[0])
    if batch_size is not None:
        nd = batch_size*(len(X)/batch_size)
        X = X[:nd]
        y = y[:nd]
    return X, y

def load_data(args, train_prop=0.9):
    # load data, make piano rolls
    d = cPickle.load(open(args.train_file))
    D = [(pianoroll_history(song_to_pianoroll(x['song']), args.seq_length), x['playcount']) for x in d]
    D = [(x,y) for x,y in D if len(x)]
    if args.n_per_song != 0:
        # min_per_song = min([len(x) for x,y in D])
        min_per_song = args.n_per_song
        print "Using {} samples per song.".format(min_per_song)
        D = sample_random_rows(D, min_per_song)

    # normalize y to be in 0..1
    ymn = np.min([y for x,y in D])
    ymx = np.max([y for x,y in D])

    # split by song into train/validation/test
    inds = np.random.permutation(len(D))
    nd = int(len(D)*train_prop)
    tr_inds = inds[:nd]
    te_inds = inds[nd::2]
    va_inds = inds[nd+1::2]
    Xtr, ytr = get_data_block(D, tr_inds, (ymn, ymx),
        args.batch_size)
    Xva, yva = get_data_block(D, va_inds, (ymn, ymx),
        args.batch_size)
    Xte, yte = get_data_block(D, te_inds, (ymn, ymx),
        args.batch_size)
    return Xtr, ytr, Xva, yva, Xte, yte

def get_model(args):
    n_filters = 64
    kernel_size = 2
    pool_size = 2
    hidden_dim = 2

    X = Input(batch_shape=(args.batch_size,
        args.seq_length, args.original_dim), name='X')
    # X.shape == [100, S, 128]

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
    y = Dense(1, activation='sigmoid', name='y')(zf)

    model = Model(X, y)
    model.compile(optimizer=args.optimizer,
        loss='binary_crossentropy')
    return model

def train(args):
    Xtr, ytr, Xva, yva, _, _ = load_data(args)
    print "Training X ({}) and y ({})".format(Xtr.shape, ytr.shape)

    1/0
    import utils.midi_utils
    pm = utils.midi_utils.roll_to_pm(100.*Xtr[0].T, fs=12.)
    pm.write('../data/samples/tmp1.mid')

    args.original_dim = Xtr.shape[-1]
    model = get_model(args)
    save_model_in_pieces(model, args)
    history = model.fit(Xtr, ytr,
        shuffle=True,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        callbacks=get_callbacks(args),
        validation_data=(Xva, yva))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
        help='tag for current run')
    parser.add_argument('--num_epochs', type=int, default=200,
        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=100,
        help='batch size')
    parser.add_argument('--seq_length', type=int, default=8,
        help='seq length')
    parser.add_argument('--n_per_song', type=int, default=50,
        help='samples per song')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--optimizer', type=str,
        default='adam', help='optimizer name')
    parser.add_argument('--train_file', type=str,
        default='../data/beatles/songs-20.pickle',
        help='file of training data (.pickle)')
    parser.add_argument('--model_dir', type=str,
        default='../data/models')
    args = parser.parse_args()
    train(args)
