import argparse
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from model import load_model
from utils.midi_utils import roll_to_pm

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

def main(args):
    model, margs = load_model(args.model_file, do_opt=True)    

    output_index = -1
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
    P[:,60] = val
    P[:,64] = val
    P[:,67] = val
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str,
        default='../data/models/tmp5.h5',
        help='file with model weights (.h5)')
    args = parser.parse_args()
    main(args)
