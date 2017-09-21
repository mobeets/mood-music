import cPickle
import numpy as np
import matplotlib.ticker as ticker
import seaborn as sns; sns.set()

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
	songfile = '../data/beatles/songs-all.pickle'
	main(songfile)
