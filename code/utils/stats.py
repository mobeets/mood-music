import os.path
import json
import joblib
import fnmatch
from utils.midi_utils import compute_statistics, midifile_roundtrip

def run_all_midifiles(midifiles, inner_fcn=None, do_parallel=True, njobs=5):
    if inner_fcn is None:
        raise Exception("Error: You must specify inner_fcn.")
    if do_parallel:
        stats = joblib.Parallel(n_jobs=njobs, verbose=0)(
            joblib.delayed(inner_fcn)(midi_file, key)
            for key, midi_file in midifiles.iteritems())
    else:
        stats = [inner_fcn(midi_file, key)
            for key, midi_file in midifiles.iteritems()]
    # When an error occurred, None will be returned; filter those out.
    stats = [s for s in stats if s is not None]
    return stats

def get_all_midifiles(mididir, infofile, maxn=None):
    """
    only keeps midifiles where we have info about them
    """
    info = json.load(open(infofile))
    midifiles = {}
    i = 0
    for root, dirs, files in os.walk(mididir):
        for infile in fnmatch.filter(files, '*.mid'):
            midifile = os.path.join(root, infile)
            nm = midifile.split('/')[-2]
            if nm not in midifiles and nm in info and 'xid' in info[nm]:
                midifiles[nm] = midifile
                i += 1
                if maxn is not None and i > maxn:
                    return midifiles
    return midifiles

def save_stats(outfile, stats):
    json.dump(stats, open(outfile, 'w'))

def make_stats(midifiles, outfile):
    print 'Found {} midifiles to process.'.format(len(midifiles))
    
    stats = run_all_midifiles(midifiles, inner_fcn=compute_statistics)
    print 'Created {} stats.'.format(len(stats)) 
    save_stats(outfile, stats)
    return stats

def make_midis(midifiles, outdir):
    print 'Found {} midifiles to process.'.format(len(midifiles))
    
    outf = lambda y: os.path.join(outdir, y + '.mid')
    pairs = dict((outf(key), midifile) for key, midifile in midifiles.iteritems())
    stats = run_all_midifiles(pairs, inner_fcn=midifile_roundtrip)

if __name__ == '__main__':
    mididir = os.path.join('data', 'lmd_matched')
    infofile = os.path.join('data', 'lmd_gracenote_metadata.json')    
    midifiles = get_all_midifiles(mididir, infofile, maxn=None)

    # keys = midifiles.keys()[25000:]
    # midifiles = dict((nm,midifiles[nm]) for nm in keys)
    # outfile = os.path.join('data', 'stats-30000.json')
    # make_stats(midifiles, outfile)

    # outdir = os.path.join('data', 'samples')
    # make_midis(midifiles, outdir)
