import os
import fnmatch
import json
from progress.bar import Bar
import pretty_midi
from unidecode import unidecode
from utils.detect_key import pm_to_key
from utils.emo_beatles import get_songname

def get_key(infile):
    try:
        pm = pretty_midi.PrettyMIDI(infile)
        return pm_to_key(pm)
    except Exception as e:
        print infile, str(e)

def save_checkpoint(outfile, info):
    json.dump(info, open(outfile, 'w'))

def main(outfile):
    info = {}
    i = 0
    bar = Bar('Processing', max=226, suffix='%(index)d completed')
    artist = 'The Beatles'
    for root, dirs, files in os.walk('data/beatles'):
        for infile in fnmatch.filter(files, '*.mid'):
            songname = get_songname(infile)
            obj = {'artist': artist, 'songname': songname}
            obj['key'] = get_key(os.path.join(root, infile))
            info[infile] = obj
            bar.next()
    bar.finish()
    save_checkpoint(outfile, info)

if __name__ == '__main__':
    outfile = '../data/beatles/keys.json'
    main(outfile)
