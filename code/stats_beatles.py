import os.path
import json
import glob
from utils.stats import get_all_midifiles, make_stats
from utils.mood import mood_to_tuple

def combine(infiles, outfile):
	ds = dict((k, json.load(open(infiles[k]))) for k in infiles)
	makeobj = lambda k: dict((kc, ds[kc].get(k, None)) for kc in infiles)
	songfiles = ds['info'].keys()
	d = dict((k, makeobj(k)) for k in songfiles)
	json.dump(d, open(outfile, 'w'))

def print_summary(metafile):
	info = json.load(open(metafile))
	for key in info:
		print '============'
		print key
		print '------------'
		for k in info[key]:
			if not info[key][k]:
				continue
			print '---{}---'.format(k.upper())
			for ki in info[key][k]:
				print ki, info[key][k][ki]

def print_filter(metafile):
	info = json.load(open(metafile))
	for key in info:
		pass

def make_all_stats(infofile, statsfile):
	mididir = '../data/beatles'
	midifiles = glob.glob(os.path.join(mididir, '*.mid'))
	midifiles = dict((os.path.split(m)[1], m) for m in midifiles)
	stats = make_stats(midifiles, statsfile)
	stats = dict((s['identifier'], s) for s in stats)
	json.dump(stats, open(statsfile, 'w'))

if __name__ == '__main__':
	infofile = '../data/beatles/info.json'
	statsfile = '../data/beatles/stats.json'
	keyfile = '../data/beatles/keys.json'
	metafile = '../data/beatles/meta.json'
	# make_all_stats(infofile, statsfile)
	# combine({'info': infofile, 'stats': statsfile, 'key': keyfile}, metafile)
	# print_summary(metafile)
	print_filter(metafile)
