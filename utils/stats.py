import os.path
import json
import joblib
import fnmatch
from utils.midi_utils import compute_statistics, midifile_roundtrip

def run_all_midifiles(midifiles, inner_fcn=None, do_parallel=True):
    if inner_fcn is None:
        raise Exception("Error: You must specify inner_fcn.")
    if do_parallel:
        stats = joblib.Parallel(n_jobs=10, verbose=0)(
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

def make_stats(midifiles):
    print 'Found {} midifiles to process.'.format(len(midifiles))
    
    stats = run_all_midifiles(midifiles, inner_fcn=compute_statistics)
    print 'Created {} stats.'.format(len(stats))
    outfile = os.path.join('data', 'stats.json')
    save_stats(outfile, stats)

def make_midis(midifiles):
    print 'Found {} midifiles to process.'.format(len(midifiles))
    
    outf = lambda y: os.path.join('data', 'samples', y + '.mid')
    pairs = dict((outf(key), midifile) for key, midifile in midifiles.iteritems())
    stats = run_all_midifiles(pairs, inner_fcn=midifile_roundtrip)

if __name__ == '__main__':
    mididir = os.path.join('data', 'lmd_matched')
    infofile = os.path.join('data', 'md5_to_info_all-v8.json')    
    midifiles = get_all_midifiles(mididir, infofile, maxn=None)

    make_stats(midifiles)
    # make_midis(midifiles)
