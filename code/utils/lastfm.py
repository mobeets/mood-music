import json
import urllib2
import pylast

LASTFM_KEY = "df9c0a4b3045740595588b0cde40e60f"
LASTFM_SECRET = "cb162a316185ba30a6b1e8b960793d8c"

track_find_url = "http://ws.audioscrobbler.com/2.0/?method=track.search&track={songname}&artist=beatles&api_key={api_key}&format=json"

track_info_url = "http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={api_key}&artist=beatles&track={songname}&format=json"

def get_track(query):
	try:
		x = urllib2.urlopen(track_find_url.format(api_key=LASTFM_KEY, songname="Maxwell"))
		d = json.loads(x.readlines()[0])
		return d['results']['trackmatches']['track'][0]['name']
	except Exception, e:
		return None

def get_plays(songname):
	try:
		x = urllib2.urlopen(track_info_url.format(api_key=LASTFM_KEY, songname=urllib2.quote(songname)))
		d = json.loads(x.readlines()[0])
		return int(d['track']['playcount'])
	except Exception, e:
		print "ERROR ({}): {}".format(songname, str(e))
		return None

def get_playcounts(metafile, outfile):
	d = json.load(open(metafile))
	for x in d:
		name = d[x]['info']['track_title']
		playcount = get_plays(name)
		if playcount is None:
			songname = get_track(name)
			if songname is not None:
				playcount = get_plays(songname)
		d[x]['playcount'] = playcount
		print name, d[x]['playcount']
	json.dump(d, open(outfile, 'w'))

if __name__ == '__main__':
	metafile = '../data/beatles/meta/meta.json'
	outfile = '../data/beatles/meta/meta1.json'
	get_playcounts(metafile, outfile)
