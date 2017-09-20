
def mood_to_tuple(label):
    label = label.lower()
    mood = {'peaceful': (-2, 2),
        'tender': (-2, 1),
        'sentimental': (-2, 0),
        'melancholy': (-2, -1),
        'somber': (-2, -2),
        'easygoing': (-1, 2),
        'romantic': (-1, 1),
        'sophisticated': (-1, 0),
        'cool': (-1, -1),
        'gritty': (-1, -2),
        'upbeat': (0, 2),
        'empowering': (0, 1),
        'sensual': (0, 0),
        'yearning': (0, -1),
        'serious': (0, -2),
        'lively': (1, 2),
        'stirring': (1, 1),
        'fiery': (1, 0),
        'urgent': (1, -1),
        'brooding': (1, -2),
        'excited': (2, 2),
        'rowdy': (2, 1),
        'energizing': (2, 0),
        'defiant': (2, -1),
        'aggressive': (2, -2)}
    return mood[label]

def summarize_moods(infofile):
    d = json.load(open(infofile))
    counts = {}
    positivity = {}
    energy = {}
    for x in d:
        if not x['mood']:
            continue
        md = x['mood']['1']['TEXT']
        if md not in counts:
            counts[md] = 0
        nrg, pos = mood_to_tuple(md)
        if nrg not in energy:
            energy[nrg] = 0
        if pos not in positivity:
            positivity[pos] = 0
        counts[md] += 1
        energy[nrg] += 1
        positivity[pos] += 1
    print '----MOOD-------'
    for mood in counts:
        print '{}: {}'.format(mood, counts[mood])
    print '----ENERGY-------'
    for mood in energy:
        print '{}: {}'.format(mood, energy[mood])
    print '-----POSITIVITY------'
    for mood in positivity:
        print '{}: {}'.format(mood, positivity[mood])

if __name__ == '__main__':
    import os.path
    import json
    infofile = os.path.join('data', 'beatles', 'info.json')
    summarize_moods(infofile)
