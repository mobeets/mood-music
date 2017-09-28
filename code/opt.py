import argparse
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from model import load_model
from utils.midi_utils import roll_to_pm
from make_songs import song_to_pianoroll

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def plot_input(Xc, outfile):
    Xc = np.hstack([Xc, np.zeros((Xc.shape[0], 4))])
    Xc = Xc.reshape(Xc.shape[0], -1, 12)

    nd = np.ceil(np.sqrt(Xc.shape[0])).astype(int)
    f, axarr = plt.subplots(nd, nd)
    i = 0
    for axs in axarr:
        for ax in axs:
            if i < len(Xc):
                sns.heatmap(Xc[i], vmin=0, vmax=1,
                    xticklabels=False, yticklabels=False,
                    ax=ax, cbar=False)
            i += 1
    plt.savefig(outfile)

def main_search(model, margs):
    output_index = 0
    loss = K.mean(model.output[:, output_index])

    # compute the gradient of the input wrt this loss
    X = model.input # placeholder for model input
    grads = K.gradients(loss, X)[0]
    grads = normalize(grads)

    # this function returns the loss and grads given the input
    iterate = K.function([X], [loss, grads])

    # we start from a C major chord played for a full measure
    # need to push in a continuous p and pass through sigmoid
    val = 10.
    P = -val*np.ones((margs.seq_length, margs.original_dim))
    # P[:,60] = val
    # P[:,64] = val
    # P[:,67] = val
    P = P[None,:,:]

    Xc = 1/(1 + np.exp(-P[0]))
    plot_input(Xc, '../data/plots/opt-in.png')
    roll_to_pm(100*Xc.T, fs=8).write('../data/samples/opt-in.mid')

    # run gradient ascent for 20 steps
    step = 1.
    vs = []
    prev_loss_value = 0.
    loss_value = -1.
    norm = lambda x: np.sqrt(np.square(x).sum())
    while norm(prev_loss_value-loss_value) > 1e-10:
        prev_loss_value = loss_value
        loss_value, grads_value = iterate([P])
        P += grads_value * step
        vs.append((loss_value, norm(grads_value)))
        # print vs[-1]
    Xc = 1/(1 + np.exp(-P[0]))
    plot_input(Xc, '../data/plots/opt-out.png')
    roll_to_pm(100*(Xc.T.round()), fs=8).write('../data/samples/opt-out.mid')
    
def main_embed(model, margs):
    min_note = 15 # empirical assumption
    min_ascii = 33 # because unichr(32) is space
    max_ascii = 127 # otherwise can't convert to str
    part_to_word = lambda part: ''.join([str(unichr(n-min_note+min_ascii+1)) for n in part]) if len(part) else str(unichr(min_ascii))
    # c_chord = part_to_word([60, 64, 67])
    # ind = margs.word_index[c_chord]
    silence = part_to_word([])
    ind = margs.word_index[silence]
    X = np.array([ind]*margs.seq_length)[None,:]

    # import cPickle
    # d = cPickle.load(open(margs.train_file))
    # vmn = min([x['playcount'] for x in d])
    # song = [x for x in d if x['playcount'] == vmn][0]
    # print song['songname']
    # song = song['song'][:margs.seq_length]
    # inds = [margs.word_index[part_to_word(p)] for p in song]
    # inds = [ind if ind < margs.num_words else 0 for ind in inds]
    # X = np.array(inds)[None,:]
    # roll = song_to_pianoroll(song)
    # roll_to_pm(100*roll.T, fs=8).write('../data/samples/opt-in-{}-{}.mid'.format(args.prefix, args.output_index))

    maxps = []
    # for each step in the sequence,
    # choose the word that will maximize the model's prediction
    # while holding the other words fixed
    for i in range(margs.seq_length)[::-1]:
        # note: there's actually margs.num_words-1 words
        Xc = np.tile(X, (margs.num_words-1, 1))
        Xc[:,i] = np.arange(1,margs.num_words)
        ps = model.predict(Xc)[:,args.output_index]
        maxps.append(np.max(ps))
        best_word = Xc[np.argmax(ps),i]
        X[0,i] = best_word
    print maxps
    print "Maximum value reached: {}".format(np.max(ps))

    ind_to_word = dict((y,x) for x,y in margs.word_index.iteritems())
    words = [ind_to_word[i] for i in X[0]]

    check_silence = lambda part: [] if part[0] == min_note-1 else part
    part_to_note = lambda note: ord(note)-1-min_ascii+min_note
    word_to_parts = lambda word: check_silence([part_to_note(c) for c in word])
    song = [word_to_parts(word) for word in words]
    print X, words, song

    roll = song_to_pianoroll(song)
    roll_to_pm(100*roll.T, fs=8).write('../data/samples/opt-out-{}-{}.mid'.format(args.prefix, args.output_index))

def main(args):
    model, margs = load_model(args.model_file, do_opt=True)    
    if margs.do_embed:
        main_embed(model, margs)
    else:
        main_search(model, margs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str,
        default='../data/models/tmp6.h5',
        help='file with model weights (.h5)')
    parser.add_argument('--output_index', type=int, default=0,
        help='index to maximize')
    parser.add_argument('--prefix', type=str, default='',
        help='output filename prefix')
    args = parser.parse_args()
    main(args)
