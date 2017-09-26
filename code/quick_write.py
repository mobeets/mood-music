import sys
import cPickle
import numpy as np
import pretty_midi
import utils.midi_utils

def main(query, outfile='../data/samples/tmpa.mid', division=1):

	songfile = '../data/beatles/songs-clean.pickle'
	songs = cPickle.load(open(songfile, 'rb'))
	res = [(i,s['filename']) for i, s in enumerate(songs) if query.lower() in s['songname'].lower()]
	if not res:
		print "No results found."
		return

	ind, fnm = res[0]
	print ind, fnm

	fnm = '../data/beatles/raw/' + fnm

	pm = pretty_midi.PrettyMIDI(fnm)
	roll = utils.midi_utils.pmtoroll(pm, division)
	roll[roll > 0] = 80.

	bpm = pm.get_tempo_changes()[1][0]
	fs = bpm*division/60.
	pm2 = utils.midi_utils.roll_to_pm(roll, fs=fs)
	pm2.write(outfile)

if __name__ == '__main__':
	query = sys.argv[1]
	division = sys.argv[2] if len(sys.argv) > 2 else 2
	main(query, division=float(division))
