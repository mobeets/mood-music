import re
import os.path
import urllib
from BeautifulSoup import BeautifulSoup

baseurl = "http://earlybeatles.com/"
pages = ["intro.html", "meet.html", "second.html", "hardday.html", "something.html", "beatle65.html", "early.html", "beatle6.html", "help.html", "rubber.html", "yesterday.html", "revolver.html", "sgtpeppers.html", "magical.html", "white.html", "yellow.html", "abbey.html", "heyjude.html", "letitbe.html", "info1.html"]

def get_all_midi_urls(url):
    midiurls = []
    html_page = urllib.urlopen(url)
    soup = BeautifulSoup(html_page)
    for link in soup.findAll('a'):
        url = link.get('href')
        if '.mid' in url:
            midiurls.append(url)
    return midiurls

def main():
    outdir = "data/beatles"
    for page in pages:
        print "Searching {}...".format(page)
        urls = get_all_midi_urls(os.path.join(baseurl, page))
        for url in urls:
            urlc = os.path.join(baseurl, url)
            name = urlc.split('/')[-1].replace('.mid', '').strip()
            outfile = os.path.join(outdir, name + '.mid')
            print "Downloading {}.".format(name)
            urllib.urlretrieve(urlc, outfile)

if __name__ == '__main__':
    main()
