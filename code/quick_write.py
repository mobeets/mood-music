import sys
import cPickle
import pretty_midi
import utils.midi_utils

def main(query, outfile='../data/samples/tmpa.mid', scale=1/10.):

	songfile = '../data/beatles/songs-all.pickle'
	songs = cPickle.load(open(songfile, 'rb'))
	res = [(i,s['name']) for i, s in enumerate(songs) if query.lower() in s['name'].lower()]
	if not res:
		print "No results found."
		return

	ind, fnm = res[0]
	print ind, fnm

	fnm = '../data/beatles/raw/' + fnm
	pm = pretty_midi.PrettyMIDI(fnm)
	bpm = pm.get_tempo_changes()[1][0]
	print bpm

	roll = utils.midi_utils.get_clean_pianoroll(fnm,
		bpm_scale=scale)
	pm2 = utils.midi_utils.roll_to_pm(roll, fs=bpm*scale)
	pm2.write(outfile)

if __name__ == '__main__':
	main(sys.argv[1])
