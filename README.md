
__Preprocessing__:

1. Download "LMD-matched" and "LMD-matched-metadata" from the [Lakh MIDI Dataset](http://colinraffel.com/projects/lmd/)
2. Find the mood of each song in the metadata by using [pygn](https://github.com/cweichen/pygn) to query the [Gracenote API](https://developer.gracenote.com/web-api)
3. Convert songs to piano roll notation, and tranpose into C major or C minor
