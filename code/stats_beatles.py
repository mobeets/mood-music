import os.path
import json
import glob
from utils.stats import get_all_midifiles, make_stats
from utils.mood import mood_to_tuple
from emo_beatles import get_songname

def combine(infiles, outfile):
	ds = dict((k, json.load(open(infiles[k]))) for k in infiles)
	makeobj = lambda k: dict((kc, ds[kc].get(k, None)) for kc in infiles)
	songfiles = ds['info'].keys()
	d = dict((k, makeobj(k)) for k in songfiles)
	json.dump(d, open(outfile, 'w'))

def print_summary(metafile):
	info = json.load(open(metafile))
	for song in info:
		print '============'
		print song
		print '------------'
		for k in info[song]:
			if not info[song][k]:
				continue
			print '---{}---'.format(k.upper())
			for ki in info[song][k]:
				print ki, info[song][k][ki]

def print_filter(metafile):
	info = json.load(open(metafile))
	posd = {}
	nrgd = {}
	alld = {}
	for song in info:
		songname = get_songname(song)
		mood = info[song]['info']['mood']
		if not mood:
			continue
		mood = mood['1']['TEXT']
		nrg, pos = mood_to_tuple(mood)
		pos = 'dark' if pos <=0 else 'positive'
		nrg = 'calm' if nrg <= 0 else 'energetic'
		key = info[song]['key']['key']
		if key is None:
			continue
		mode = 'major' if 'major' in key else 'minor'
		bot = mode + ' ' + pos + ' ' + nrg
		pos = mode + ' ' + pos
		nrg = mode + ' ' + nrg
		if nrg not in nrgd:
			nrgd[nrg] = []
		if pos not in posd:
			posd[pos] = []
		if bot not in alld:
			alld[bot] = []
		nrgd[nrg].append(songname)
		posd[pos].append(songname)
		alld[bot].append(songname)
	print '----MOOD----'
	for k in alld:
		print k, len(alld[k])#, alld[k][:2]
	print '----ENERGY----'
	for k in nrgd:
		print k, len(nrgd[k])#, nrgd[k][:2]
	print '----POSITIVITY----'
	for k in posd:
		print k, len(posd[k])#, posd[k][:2]

def make_all_stats(infofile, statsfile):
	mididir = '../data/beatles'
	midifiles = glob.glob(os.path.join(mididir, '*.mid'))
	midifiles = dict((os.path.split(m)[1], m) for m in midifiles)
	stats = make_stats(midifiles, statsfile)
	stats = dict((s['identifier'], s) for s in stats)
	json.dump(stats, open(statsfile, 'w'))

if __name__ == '__main__':
	infofile = '../data/beatles/meta/info.json'
	statsfile = '../data/beatles/meta/stats.json'
	keyfile = '../data/beatles/meta/keys.json'
	metafile = '../data/beatles/meta/meta.json'
	# make_all_stats(infofile, statsfile)
	# combine({'info': infofile, 'stats': statsfile, 'key': keyfile}, metafile)
	# print_summary(metafile)
	print_filter(metafile)
