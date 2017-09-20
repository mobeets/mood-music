import numpy as np

cmaj = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
cmin = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
offsets = {"C": 0, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}
lkp = dict((offsets[nm], nm) for nm in offsets)

def krumhansl_analyze(query, template):
    """
    Compares two vectors to give a similarity, based on paper by krumhansl
    inputs:
        a query vector representing the total duration of pitches in the song
        a template vector representing the common prevalance of pitches in the key
    outputs: a score for how similar the key is to the song
    """
    return np.corrcoef(query, template)[0,1]

def analyze_key(query):
    """
    Gets the key from a file, using the binning method
    inputs:
        a list of the duration of each pitch in the song, from C to B
    outputs:
        the key of the input (i.e., has maximum correlation with template)
    """
    nkeys = len(cmaj)
    maj_scs = [krumhansl_analyze(query, np.roll(cmaj, i)) for i in xrange(nkeys)]
    min_scs = [krumhansl_analyze(query, np.roll(cmin, i)) for i in xrange(nkeys)]
    # for i,v in enumerate(maj_scs):
    #     print lkp[i], v
    # for i,v in enumerate(min_scs):
    #     print lkp[i].lower(), v
    maj_key_ind = np.argmax(maj_scs)
    min_key_ind = np.argmax(min_scs)
    if maj_scs[maj_key_ind] > min_scs[min_key_ind]:
        return lkp[maj_key_ind] + ' major'
    else:
        return lkp[min_key_ind] + ' minor'

def pm_to_key(pm):
    cs = pm.get_pitch_class_histogram(use_duration=True)
    return analyze_key(cs)

def pianoroll_to_key(roll):
    """
    assumes roll is [128 x nt]

    WARNING: is this rolled correctly? i.e., starts with C and not A?
    """
    chroma_matrix = np.zeros((12,))
    for note in range(12):
        chroma_matrix[note] = roll[note::12].sum(axis=0).sum(axis=-1)
    return analyze_key(chroma_matrix)
