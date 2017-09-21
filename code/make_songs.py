import os.path
import json
import cPickle
import numpy as np
import joblib
from music21.key import Key
from music21.pitch import Pitch
from utils.midi_utils import get_clean_pianoroll

def get_offset_from_key(key):
	tonic, mode = key.split(' ')
	tonic = tonic.replace('b', '-')
	if mode == 'major':
		k = Key(tonic)
	else:
		k = Key(tonic.lower()).relative
	return Pitch('C').midi - k.getTonic().midi

def pianoroll_to_song(roll):
    return [(np.where(r)[0]).tolist() for r in roll]

def song_to_pianoroll(song, offset=0):
    """
    song = [(60, 72, 79, 88), (72, 79, 88), (67, 70, 76, 84), ...]
    """
    rolls = []
    for notes in song:
        roll = np.zeros(128)
        roll[[n-offset for n in notes]] = 1.
        rolls.append(roll)
    return np.vstack(rolls)

def make_song(name, song, mididir, do_key_shift=True, bpm_scale=1/10.):
	key = song['key']['key']
	if do_key_shift:
		if key is None:
			return None
		offset = get_offset_from_key(key)
	else:
		offset = 0
	midifile = os.path.join(mididir, name)
	roll = get_clean_pianoroll(midifile, offset, bpm_scale=bpm_scale).T
	song = pianoroll_to_song(roll)
	return {'song': song, 'key': key, 'key_mode': 'major' in key, 'name': name, 'offset': offset}

def make_all_songs(metafile, mididir, songfile, do_key_shift=True):
	info = json.load(open(metafile))
	# info = dict((x,info[x]) for x in info.keys()[:20])
	print 'Found {} midifiles.'.format(len(info))
	songs = joblib.Parallel(n_jobs=5, verbose=0)(joblib.delayed(make_song)(name, info[name], mididir, do_key_shift) for name in info)
	songs = [x for x in songs if x is not None]
	print 'Created {} songs.'.format(len(songs))
	cPickle.dump(songs, open(songfile, 'w'))

if __name__ == '__main__':
	mididir = '../data/beatles/raw'
	metafile = '../data/beatles/meta/meta.json'
	songfile = '../data/beatles/songs-all.pickle'
	make_all_songs(metafile, mididir, songfile)
