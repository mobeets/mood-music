
__Preprocessing__:

1. Download "LMD-matched" and "LMD-matched-metadata" from the [Lakh MIDI Dataset](http://colinraffel.com/projects/lmd/)
2. Find the mood of each song in the metadata by using [pygn](https://github.com/cweichen/pygn) to query the [Gracenote API](https://developer.gracenote.com/web-api)
3. Discretize midi data into sixteenth notes, convert to piano-roll notation (binary matrix of [nsamples x npitches]), and tranpose to C major or C minor.
