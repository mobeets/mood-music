import os.path
import json
import cPickle
import numpy as np
import joblib
from music21.key import Key
from music21.pitch import Pitch
import pretty_midi
from utils.midi_utils import get_clean_pianoroll, pmtoroll
from utils.mood import mood_to_tuple

def get_offset_from_key(key):
	tonic, mode = key.split(' ')
	tonic = tonic.replace('b', '-')
	if mode == 'major':
		k = Key(tonic)
	else:
		k = Key(tonic.lower()).relative
	return Pitch('C').midi - k.getTonic().midi

def pianoroll_to_song(roll, trim_flank_silence=False):
    song = [(np.where(r)[0]).tolist() for r in roll]
    if trim_flank_silence:
    	# trim silence at beginning or end
    	if len(song) == 0 or len([x for x in song if len(x)]) == 0:
    		return []
		inds = [i for i,x in enumerate(song) if len(x) > 0]
		song = song[inds[0]:(inds[-1]+1)]
	return song

def song_to_pianoroll(song, offset=0, trim_flank_silence=False):
    """
    song = [(60, 72, 79, 88), (72, 79, 88), (67, 70, 76, 84), ...]
    """
    rolls = []
    started = not trim_flank_silence
    for notes in song:
    	if not started and len(notes) == 0:
    		continue
    	else:
    		started = True
        roll = np.zeros(128)
        roll[[n-offset for n in notes]] = 1.
        rolls.append(roll)
    return np.vstack(rolls)

def pianoroll_history(roll, seq_length, stride_length=1):
	if len(roll) < seq_length:
		return []
	rs = []
	for i in np.arange(0, len(roll)-seq_length+1, stride_length):
		rs.append(roll[i:i+seq_length])
	rs = np.dstack(rs)
	return np.transpose(rs, (2,0,1))

def get_mood(x):
	mood = x['mood']
	if not mood:
		return None, None, None
	mood = mood['1']['TEXT']
	nrg, pos = mood_to_tuple(mood)
	return mood, nrg, pos

def make_song(filename, song, mididir, do_key_shift=True, division=1):
	key = song['key']['key']
	songname = song['info']['track_title']
	playcount = song['playcount']
	mood, mood_energy, mood_valence = get_mood(song['info'])
	if do_key_shift:
		if key is None:
			return None
		offset = get_offset_from_key(key)
	else:
		offset = 0
	midifile = os.path.join(mididir, filename)
	# roll = get_clean_pianoroll(midifile, offset, division=division).T
	pm = pretty_midi.PrettyMIDI(midifile)
	roll = pmtoroll(pm, division=division, offset=offset)
	if roll is None:
		return None
	song = pianoroll_to_song(roll.T, trim_flank_silence=True)
	return {'songname': songname, 'song': song, 'key': key,
		'key_mode': 'major' in key, 'filename': filename,
		'offset': offset, 'mood': mood,
		'mood_energy': mood_energy, 'mood_valence': mood_valence,
		'playcount': playcount}

def make_all_songs(metafile, mididir, songfile, do_key_shift=True, division=4, do_parallel=True):
	info = json.load(open(metafile))
	# info = dict((x,info[x]) for x in info.keys()[:20])
	# info = dict((x,info[x]) for x in info.keys() if x == u'11B-1__Within_You_Without_You.mid')
	print 'Found {} midifiles.'.format(len(info))
	if do_parallel:
		songs = joblib.Parallel(n_jobs=5, verbose=0)(joblib.delayed(make_song)(filename, info[filename], mididir, do_key_shift, division) for filename in info)
	else:
		songs = [make_song(filename, info[filename], mididir, do_key_shift, division) for filename in info]
	# keep only unique, non-None songs
	names = []
	allsongs = []
	for x in songs:
		if x is None:
			continue
		if x['songname'] not in names and len(x['song']) > 0:
			names.append(x['songname'])
			allsongs.append(x)
	songs = allsongs
	print 'Created {} songs.'.format(len(songs))
	cPickle.dump(songs, open(songfile, 'w'))

if __name__ == '__main__':
	mididir = '../data/beatles/raw'
	metafile = '../data/beatles/meta/meta.json'
	songfile = '../data/beatles/songs-sixteenths.pickle'
	make_all_songs(metafile, mididir, songfile)
