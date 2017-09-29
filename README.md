__Data__: MIDI data for songs in the Million Songs Dataset can be obtained by downloading "LMD-matched" and "LMD-matched-metadata" from the [Lakh MIDI Dataset](http://colinraffel.com/projects/lmd/).

__Preprocessing__:

1. Find the mood of each song using [pygn](https://github.com/cweichen/pygn) to query the [Gracenote API](https://developer.gracenote.com/web-api)
2. Discretize midi data into sixteenth notes, convert to piano-roll notation (binary matrix of [nsamples x npitches]), and tranpose to C major or C minor.
