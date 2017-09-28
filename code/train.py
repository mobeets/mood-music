import cPickle
import argparse
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from utils.weightnorm import data_based_init
from utils.model_utils import get_callbacks, save_model_in_pieces
from make_songs import song_to_pianoroll, pianoroll_history
from model import get_model, get_embed_model

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

def seq_history(seq, seq_length, stride_length=1):
    if len(seq) < seq_length:
        return []
    rs = []
    for i in np.arange(0, len(seq)-seq_length+1, stride_length):
        rs.append(seq[i:i+seq_length])
    return np.vstack(rs)

def embed_data(d, args):
    """
    converts songs to sentences of numbers using tokenizer
    note:
        - unichr(32) is used for spaces, to delimit words
        - empty part "[]" is mapped to unichr(33)
        - parts with notes are mapped in unichr(34)...unichr(127)
    """
    print "Embedding..."
    min_note = 15 # empirical assumption
    min_ascii = 33 # because unichr(32) is space
    max_ascii = 127 # otherwise can't convert to str
    assert min([min([n for p in x['song'] for n in p]) for x in d]) >= min_note
    assert max([max([n for p in x['song'] for n in p]) for x in d]) - min_note+min_ascii+1 <= max_ascii

    part_to_word = lambda part: ''.join([str(unichr(n-min_note+min_ascii+1)) for n in part]) if len(part) else str(unichr(min_ascii))
    song_to_sentence = lambda song: ' '.join([part_to_word(p) for p in song])
    skip_start_silence = lambda song: song[next(i for i,x in enumerate(song) if len(x)):]
    songs = [song_to_sentence(skip_start_silence(x['song'])) for x in d]
    Y = [x[args.yname] for x in d]
    print 'Found {} songs.'.format(len(songs))

    args.num_words = 20000
    tokenizer = Tokenizer(num_words=args.num_words,
        filters='', lower=False, split=' ')
    tokenizer.fit_on_texts(songs)
    sequences = tokenizer.texts_to_sequences(songs)
    word_index = tokenizer.word_index
    args.word_index = word_index
    print 'Found {} unique tokens.'.format(len(word_index))
    print 'Using {} tokens.'.format(args.num_words)

    sequences = [seq_history(seq, args.seq_length, args.stride_length) for seq in sequences]
    return zip(sequences, Y)

def load_data(args):
    # load data, make piano rolls
    d = cPickle.load(open(args.train_file))
    if args.do_embed:
        D = embed_data(d, args)
    else:
        D = [(pianoroll_history(song_to_pianoroll(x['song'], trim_flank_silence=True), args.seq_length, args.stride_length), x[args.yname]) for x in d]
    D = [(x,y) for x,y in D if len(x) and y is not None]
    if args.n_per_song != 0:
        min_per_song = int(np.median([len(x) for x,y in D]))
        print "Median parts per song is {}.".format(min_per_song)
        # min_per_song = args.n_per_song
        args.n_per_song = min_per_song
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

    # reduce dimensionality
    if args.dim_reduce:
        print "Reducing dimensionality..."
        dmn = min([np.where(x.sum(axis=0).sum(axis=0))[0].min() for x,y in D])
        dmx = max([np.where(x.sum(axis=0).sum(axis=0))[0].max() for x,y in D])
        args.dimrange = (dmn, dmx) # save range
        print "Reduced dimensionality from {} to {}".format(D[0][0].shape[-1], dmx-dmn+1)
        D = [(x[:,:,dmn:(dmx+1)],y) for x,y in D]

    if not args.do_classify:
        # normalize y to be in 0..1
        ymn = np.min([y for x,y in D])
        ymx = np.max([y for x,y in D])
        yrng = (ymn, ymx)
    else:
        yrng = None

    # split by song into train/validation/test
    print "Splitting into train and test..."
    if args.cv_parts:
        X, y, I = get_data_block(D, np.arange(len(D)), yrng)
        inds = np.random.permutation(len(X))
        nd = int(len(X)*args.train_prop)
        nd = (nd / args.batch_size)*args.batch_size
        Itr = inds[:nd]
        Ite = inds[nd::2]
        Iva = inds[nd+1::2]
        Ite = Ite[:(len(Ite)/args.batch_size)*args.batch_size]
        Iva = Iva[:(len(Iva)/args.batch_size)*args.batch_size]
        Xtr, ytr = X[Itr], y[Itr]
        Xte, yte = X[Ite], y[Ite]
        Xva, yva = X[Iva], y[Iva]
    else:
        inds = np.random.permutation(len(D))
        nd = int(len(D)*args.train_prop)
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

    if args.do_classify:
        ys = [ytr, yte, yva]
        args.n_classes = len(np.unique(np.hstack(ys)))
        ytr = to_categorical(ytr, args.n_classes)
        yte = to_categorical(yte, args.n_classes)
        yva = to_categorical(yva, args.n_classes)

    return AttrDict({'Xtr': Xtr, 'ytr': ytr,
        'Xva': Xva, 'yva': yva, 'Xte': Xte, 'yte': yte,
        'Itr': Itr, 'Ite': Ite, 'Iva': Iva})

def plot(D, model):
    import matplotlib.pyplot as plt
    yha = model.predict(D.Xva, batch_size=args.batch_size)
    plt.plot(D.yva, yha, '.', color='0.8')
    if not args.cv_parts:
        for ind in np.unique(D.Iva):
            plt.plot(D.yva[D.Iva == ind].mean(), yha[D.Iva == ind].mean(), 'ko')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.savefig('../data/plots/yva_hat.png')
    yha2 = model.predict(D.Xtr, batch_size=args.batch_size)
    plt.clf()
    plt.plot(D.ytr, yha2, '.', color='0.8')
    if not args.cv_parts:
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

    print "Training X ({}) and y ({})".format(D.Xtr.shape, D.ytr.shape)

    args.original_dim = D.Xtr.shape[-1]
    if args.do_embed:
        model = get_embed_model(args)
    else:
        model = get_model(args)
    model_file = save_model_in_pieces(model, args)
    data_based_init(model, D.Xtr[:100])
    history = model.fit(D.Xtr, D.ytr,
        shuffle=True,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        callbacks=get_callbacks(args),
        validation_data=(D.Xva, D.yva))
    if not args.do_classify:
        model.load_weights(model_file)
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
    parser.add_argument('--train_prop', type=float, default=0.9,
        help='proportion of data used for training')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--yname', type=str,
        choices=['playcount', 'mood_valence', 'mood_energy'],
        default='playcount', help='what to fit')
    parser.add_argument("--do_classify", action="store_true",
        help="treat y as category")
    parser.add_argument("--do_jitter", action="store_true",
        help="add jitter to piano rolls")
    parser.add_argument("--dim_reduce", action="store_true",
        help="eliminate unused midi dimensions")
    parser.add_argument("--do_embed", action="store_true",
        help="embed notes instead of using pianoroll")
    parser.add_argument("--cv_parts", action="store_true",
        help="cross-validation at part-level, not song-level")
    parser.add_argument('--optimizer', type=str,
        default='adam', help='optimizer name')
    parser.add_argument('--train_file', type=str,
        default='../data/beatles/songs-16s.pickle',
        help='file of training data (.pickle)')
    parser.add_argument('--model_dir', type=str,
        default='../data/models')
    args = parser.parse_args()
    train(args)
