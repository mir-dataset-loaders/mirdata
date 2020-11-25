***************************************
************* CANTE100 ****************
***************************************

- Version 1.0. -
Nadine Kroher
Music Technology Group
Universitat Pompeu Fabra
2015


===== INTRODUCTION ======
The cante100 dataset contains 100 track taken from corpusCofla. Manual annotations include style family (10 different style families with 10 tracks each) and vocal sections. cante100 is a representative subsample of the full research corpus. All manual and automatic annotations as well as meta-information are publicly available for download. The full audio files are available on request for research purposes only. 


===== CONTACT ======
Nadine Kroher
nadine.kroher@upf.edu
José-Miguel Díaz-Báñez
dbanez@us.es


====== MANUAL ANNOTATIONS =======

*** Vocal sections ***
- manual annotation of frames containing vocals
- shouts and comments from the audience were not cinsedered vocal frames
- annotation in frames of 128samples at a sample rate of 44.1kHz; 1=vocals are present; 0=vocals are not present
- .csv file containing two coma-separated columns as follows:
<time code [s]>, <vocals (BOOL)>

*** Style family ***
- in correspondence with flamenco experts, ten style families were defined: 1. fandangos, 2. tientos & tangos, 3. soleares, 4. seguiriyas, 5. bulerías, 6. cantiñas, 7. cantes mineros, 8. cantes americanos, 9. malagueñas & granaínas, 10. tonás
- 'cante100Meta.xml' contains manual annotation for each track in
   <manual_annotation>
	<style_family></style_family>
   </manual_annotation>

====== AUTOMATIC ANNOTATIONS ======

*** Predominant melody ***
- fundamental frequency (f0) corresponding to the predominant melody
- extracted with the MELODIA[1] algorithm: minFreq = 120Hz, maxFreq = 720Hz, v = 0.2, sampleRate = 44.1kHz, windowSize = 1024samples, hopSize = 128samples
- .f0.csv file containing two comma-separated columns as follows:
  <time code [s]>, <f0 [Hz]>
- for frames in which the predominant melody is assumed not to be present, the f0 value is <=0

*** Low-level descriptors ***
- a set of low-level descriptors related to time and frequency domain characteristics extracted with the ESSENTIA[2] library: sampleRate = 44.1kHz, windowSize = 1024samples, hopSize = 512samples
- .lowlevel.csv files contianing 8 comma-separated columns as follows:
<time code[s]>, <spectral flux>, <spectral roll-off [Hz]>, <spectral complexity>, <spectral flatness>, <spectral centroid [Hz]>, <RMS>, <zero-crossing rate>

*** Mel-frequency cepstral coefficients ***
- frame-wise mel-frequency cepstral coefficients (MFCCs) extracted with the ESSENTIA[2] library: sampleRate = 44.1kHz, windowSize = 1024samples, hopSize = 512samples
- .mfccs.csv files containing 14 comma-separated columns as follows:
<time code[s]>, <mfcc1>, ..., <mfcc13>

*** Magnitude spectrum ***
- frame-wise magnitudes of 512 discrete fourier transform (DFT) bins extracted with the ESSENTIA[2] library: sampleRate = 44.1kHz, windowSize = 1024samples, hopSize = 512samples
- .spectrum.csv files containing 513 comma-separated columns as follows:
<time code[s]>, <magnitude bin1>, ..., <magnitude bin512>

*** Bark bands ***
- frame-wise energies of the bark bands extracted with the ESSENTIA[2] library: sampleRate = 44.1kHz, windowSize = 1024samples, hopSize = 512samples
- band frequencies: [0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 15500.0, 20500.0, 27000.0]
- .barkbands.csv files containing 29 comma-separated columns as follows:
<time code[s]>, <energy band 1>, ... <energy band 28>

*** Automatic transcription ***
- automatic note-level transcriptions obtained with the system described in [3]
- .mid: Standard MIDI format. To facilitat alignment with the audio track, an A4 note is inserted at the track beginning. The velocity is set to 100 for all notes.
- .notes: Text document. Three comma-separated columns as follows: 
  <note onset time [s]>, <note duration [s]>, <pitch [MIDI note]> 


===== META-DATA ANNOTATIONS =====
- XML file containing entries for all tracks
- source: anthology name, CD no. and track no. 
- editorial meta data: artist, title, style, duration min. and duration sec. 
- MusicBrainzID (see www.musicbrainz.org)
 

===== REFERENCES =====
[1] J. Salamon and E. Gómez. 2011. Melody extraction from polyphonic music signals using pitch contour characteristics,”. IEEE Transactions on Audio, Speech and Language Processing 20, 6 (2011), 1759–1770.
[2] ESSENTIA: an Audio Analysis Library for Music Information Retrieval. 
http://essentia.upf.edu
[3] N. Kroher and E. Gómez. (In peer review). Automatic Transcription of Flamenco Singing from Polyphonic Music Recordings. Submitted to the IEEE Transactions of Audio, Speech and Langauage Processing. 
