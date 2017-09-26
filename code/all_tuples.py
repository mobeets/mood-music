import json
import cPickle
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns; sns.set()

def playcount(metafile):
	d = json.load(open(metafile))
	sngs = []
	ds = {}
	for x in d:
		sng = d[x]['info']['track_title']
		if sng not in sngs:
			sngs.append(sng)
			ds[x] = d[x]
	d = ds

	vs = [d[x]['playcount'] for x in d if 'playcount' in d[x]]
	scs = [(d[x]['info']['track_title'], d[x]['playcount']) for x in d if 'playcount' in d[x]]
	scs = sorted(scs, key=lambda x: x[1], reverse=True)
	for x,s in scs[:10]:
		print '"{}": {}'.format(x, s)
	print np.median(vs), np.mean(vs)

	vs = 1.0*np.array(vs)
	
	vsc = vs - vs.min()
	vsc = vsc/vsc.max()
	ax = sns.distplot(vsc, kde=False, rug=True)
	# ax.set_yscale('log')
	ax.set_xlabel('normalized playcount')
	ax.set_ylabel('frequency')
	fig = ax.get_figure()
	fig.savefig("../data/plots/playcount-hist.png")
	return
	
	xs = np.arange(5, 100, 5)
	prcs = np.percentile(vs, xs)#.astype()
	ax = sns.regplot(prcs, xs, color='k', fit_reg=False)
	ax.set_xlabel('playcount')
	ax.set_ylabel('percentile')
	# ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
	fig = ax.get_figure()
	fig.savefig("../data/plots/playcount-prcs.png")

def tuples_in_song(song):
	xs = []
	for step in song:
		s = np.sort(step)
		for n1,n2 in zip(s[::2], s[1::2]):
			xs.append((n1, n2))
	return xs

def main(songfile, query=None):
	songs = cPickle.load(open(songfile, 'rb'))
	if query:
		songs = [s for i, s in enumerate(songs) if query.lower() in s['name'].lower()]
	print 'Found {} songs.'.format(len(songs))
	tuples = [tuples_in_song(song['song']) for song in songs]
	xs = np.vstack([t for t in tuples if len(t)])
	grid = np.zeros((xs[:,0].max()+1,xs[:,1].max()+1))
	for x,y in xs:
		grid[x,y] += 1
	# grid = grid/grid.max()
	grid = np.log(grid+1)
	# grid = (grid > 0).astype(int)
	ax = sns.heatmap(grid)#q vmin=0, vmax=1)
	ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
	ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
	ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
	ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
	fig = ax.get_figure()
	fig.savefig("../data/plots/grid-all-ystrdy.png")

if __name__ == '__main__':
	metafile = '../data/beatles/meta/meta.json'
	playcount(metafile)
	# songfile = '../data/beatles/songs-all.pickle'
	# main(songfile)
