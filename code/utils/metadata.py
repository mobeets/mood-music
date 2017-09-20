import os
import fnmatch
import json
import h5py
from progress.bar import Bar
from unidecode import unidecode
from utils import pygn

clientID = '1275500366-A7B20DB53F5F80C496FF304B17C3F748'
userID = pygn.register(clientID)

def get_artist_and_songname(infile):
    f = h5py.File(infile, 'r')
    obj = f['metadata']['songs'][0]
    return obj[9], obj[18]

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

metadir = '/Users/mobeets/Downloads/lmd_matched_h5'
infofile = os.path.join('data', 'md5_to_info_all-v15.json')
outfile = os.path.join('data', 'md5_to_info_all-v16.json')
def main():
    info = json.load(open(infofile))
    nstart = len([x for x in info if 'xid' in info[x]])
    ntotal = 21350 - (nstart-9684)
    print 'Found {} midifiles with info. {} remaining.'.format(nstart, ntotal)
    nms = []
    bar = Bar('Processing', max=ntotal,
        suffix='%(index)d completed (%(remaining)d) remaining)')
    
    i = 0
    c = 0
    # wait_for = 'TRSEKOG128F422EDC3'
    can_go = True
    # for every file in metadir, find h5 file to find artist/songname
    # then query gracenote for info on this song
    # after a while, gracenote throttles us, so just break and start over
    for root, dirs, files in os.walk(metadir):
        for infile in fnmatch.filter(files, '*.h5'):
            nm = os.path.splitext(infile)[0] # e.g., "TRNDAKX128F146EC3A"
            if not can_go and nm == wait_for:
                can_go = True
            if not can_go:
                continue
            if (nm not in info or 'xid' not in info[nm]) and nm not in nms:
                i += 1
                bar.next()
                nms.append(nm)
                if nm in info:
                    artist = info[nm]['artist_name_from_h5']
                    songname = info[nm]['song_name_from_h5']
                else:
                    h5file = os.path.join(root, infile)
                    artist, songname = get_artist_and_songname(h5file)
                    info[nm] = {'artist_name_from_h5': artist, 'song_name_from_h5': songname}
                assert 'xid' not in info[nm]
                meta = find_song(artist, songname)
                if meta is not None:
                    info[nm].update(meta)
                else:
                    c += 1
                    if c > 100:
                        save_checkpoint(outfile, info)
                        print "BREAKING at {}, {}.".format(i, nm)
                        return
                if i % 25 == 0:
                    save_checkpoint(outfile, info)
    bar.finish()
    save_checkpoint(outfile, info)

if __name__ == '__main__':
    main()
