import os
import fnmatch
import json
from progress.bar import Bar
from unidecode import unidecode
from utils import pygn

clientID = '1275500366-A7B20DB53F5F80C496FF304B17C3F748'
userID = pygn.register(clientID)

def get_artist_and_songname(filename):
    artist = 'The Beatles'
    songname = filename.replace('.mid', '').split('__')[1].replace('_', ' ').replace('---', '')
    return artist, songname

def clean(txt):
    return unidecode(txt)
    # return unidecode(txt.decode('utf-8'))

def find_song(artist, songname):
    try:
        return pygn.search(clientID=clientID, userID=userID,
            artist=clean(artist),
            track=clean(songname))
    except Exception,e:
        print str(e), artist, songname
        return None

def save_checkpoint(outfile, info):
    json.dump(info, open(outfile, 'w'))

def main(outfile):
    info = {}
    i = 0
    bar = Bar('Processing', max=226, suffix='%(index)d completed')
    for root, dirs, files in os.walk('data/beatles'):
        for infile in fnmatch.filter(files, '*.mid'):
            artist, songname = get_artist_and_songname(infile)
            meta = find_song(artist, songname)
            if meta is not None:
                bar.next()
                obj = {'artist': artist, 'songname': songname}
                meta.update(obj)
                info[infile] = meta
                i += 1
                if i % 20 == 0:
                    save_checkpoint(outfile, info)
    bar.finish()
    save_checkpoint(outfile, info)

if __name__ == '__main__':
    outfile = '../data/beatles/info.json'
    # main(outfile)
