import numpy as np
import pretty_midi
from utils.detect_key import analyze_key

def compute_statistics(midi_file, identifier=None):
    """
    Given a path to a MIDI file, compute a dictionary of statistics about it
    
    Parameters
    ----------
    midi_file : str
        Path to a MIDI file.
    
    Returns
    -------
    statistics : dict
        Dictionary reporting the values for different events in the file.
    """
    # Extract informative events from the MIDI file
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        return {
            'identifier': identifier,
            'midi_file': midi_file,
            'n_instruments': len(pm.instruments),
            'resolution': pm.resolution,
            'program_numbers': [i.program for i in pm.instruments if not i.is_drum],
            'key_numbers': [k.key_number for k in pm.key_signature_changes],
            'bpm_changes': list(pm.get_tempo_changes()[0]),
            'bpms': list(pm.get_tempo_changes()[1]),
            'time_signature_changes': [(x.numerator, x.denominator, x.time) for x in pm.time_signature_changes],
            'end_time': pm.get_end_time()}
    # ignore exceptions for a clean presentation
    except Exception as e:
        print str(e)

def pm_change_key(pm, offset):
    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch += offset
    return pm

def pmtoroll(pm, division=1, offset=0, enforce_unique_tempo=True):
    # find start and end (in seconds)
    start = min([min(i.notes, key=lambda x: x.start).start for i in pm.instruments])
    end = max([max(i.notes, key=lambda x: x.end).end for i in pm.instruments])

    # find tempo to determine temporal bin size
    tempo_change_times, tempi = pm.get_tempo_changes()
    delta = (60/tempi[0])/float(division)
    nbins = int(np.ceil((end-start)/delta))+1
    if enforce_unique_tempo and len(tempo_change_times) > 1:
        true_end = tempo_change_times[1]
        nbins = int(np.round((true_end-start)/delta))
        if nbins < 0:
            return None

    R = np.zeros([128, nbins])
    for inst in pm.instruments:
        for note in inst.notes:
            cstart = int(np.round((note.start-start)/delta))
            cend = int(np.round((note.end-start)/delta))
            cend = max(cend, cstart+1)
            if cend >= nbins:
                continue
            note.pitch += offset
            if note.pitch > R.shape[0]:
                print "Error, offset {} too large.".format(offset)
            pvel = R[note.pitch, cstart:cend]
            cvel = note.velocity*np.ones(cend-cstart)
            vel = np.vstack([pvel, cvel]).max(axis=0)
            R[note.pitch, cstart:cend] = vel
    # don't allow velocities to be above maximum
    R[R > 127.] = 127.
    return R

def get_clean_pianoroll(midi_file, offset=0, division=1):
    """
    division = number of notes to divide a quarter note
        - e.g., 1 -> quarter notes, 2 -> eighth notes, ...

    todo: need to discretize time before piano_roll hits it...
    try pm.adjust_times(original_times, new_times)
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_file)
        # just keep the first part up to the first tempo change
        if offset != 0:
            pm = pm_change_key(pm, offset)
        bpm = pm.get_tempo_changes()[1][0]
        bpm = bpm*(division/60.)
        start_times = list(pm.get_tempo_changes()[0])
        if len(start_times) > 1:
            end_time = start_times[1]
        else:
            end_time = pm.get_end_time()
        ntimes = len(np.arange(0, end_time, 1./bpm))
        roll = pm.get_piano_roll(fs=bpm)[:,:ntimes]
        roll[roll > 127.] = 127.
        return roll
    except Exception as e:
        print 'ERROR: {}'.format(e)

def roll_to_pm(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')
    piano_roll[piano_roll > 127.] = 127.

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

def midifile_roundtrip(midi_file, outfile, do_clean=True):
    if do_clean:
        sample = get_clean_pianoroll(midi_file)
        bpm = 120.
        pm = pretty_midi.PrettyMIDI(midi_file)
    else:
        pm = pretty_midi.PrettyMIDI(midi_file)
        bpm = pm.get_tempo_changes()[1][0]
        sample = pm.get_piano_roll(fs=bpm)
    if sample is None:
        return
    pm.write(outfile.replace('.mid', '_og.mid'))
    roll_to_pm(sample, fs=bpm).write(outfile)
