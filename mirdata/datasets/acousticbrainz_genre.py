# -*- coding: utf-8 -*-
"""Acoustic Brainz Genre dataset
The AcousticBrainz Genre Dataset consists of four datasets of genre annotations and music features extracted from audio
suited for evaluation of hierarchical multi-label genre classification systems.

Description about the music features can be found here: https://essentia.upf.edu/streaming_extractor_music.html

The datasets are used within the MediaEval AcousticBrainz Genre Task. The task is focused on content-based music
genre recognition using genre annotations from multiple sources and large-scale music features data available in the
AcousticBrainz database. The goal of our task is to explore how the same music pieces can be annotated differently by
different communities following different genre taxonomies, and how this should be addressed by content-based genre r
ecognition systems.

We provide four datasets containing genre and subgenre annotations extracted from four different online metadata sources:

AllMusic and Discogs are based on editorial metadata databases maintained by music experts and enthusiasts. These sources contain explicit genre/subgenre annotations of music releases (albums) following a predefined genre namespace and taxonomy. We propagated release-level annotations to recordings (tracks) in AcousticBrainz to build the datasets.

Lastfm and Tagtraum are based on collaborative music tagging platforms with large amounts of genre labels provided by their users for music recordings (tracks). We have automatically inferred a genre/subgenre taxonomy and annotations from these labels.

For details on format and contents, please refer to the data webpage.

Note, that the AllMusic ground-truth annotations are distributed separately at https://zenodo.org/record/2554044.

A size comparative between different datasets of Acoustic brainz Genre:

Citation

If you use the MediaEval AcousticBrainz Genre dataset or part of it, please cite our ISMIR 2019 overview paper:

Bogdanov, D., Porter A., Schreiber H., Urbano J., & Oramas S. (2019).
The AcousticBrainz Genre Dataset: Multi-Source, Multi-Level, Multi-Label, and Large-Scale.
20th International Society for Music Information Retrieval Conference (ISMIR 2019).


Acknowledgements

This work is partially supported by the European Union’s Horizon 2020 research and innovation programme under grant agreement No 688382 AudioCommons.
"""

import json
import os
import shutil
import urllib

from mirdata import download_utils, core
from mirdata import jams_utils
from mirdata import utils


BIBTEX = """
abadabia
"""
REMOTES = {
    'validation-01': download_utils.RemoteFileMetadata(
        filename='acousticbrainz-mediaeval-features-validation-01234567.tar.bz2',
        url='https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-validation-01234567.tar.bz2?download=1',
        checksum='f21f9c5e398713139cca9790b656faf9',
        destination_dir='temp',
    ),
    'validation-89': download_utils.RemoteFileMetadata(
        filename='acousticbrainz-mediaeval-features-validation-89abcdef.tar.bz2',
        url='https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-validation-89abcdef.tar.bz2?download=1',
        checksum='34f47394ac6d8face4399f48e2b98ebe',
        destination_dir='temp',
    ),
    'train-01': download_utils.RemoteFileMetadata(
        filename='acousticbrainz-mediaeval-features--train-01.tar.bz2',
        url='https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features--train-01.tar.bz2?download=1',
        checksum='db7157b5112022d609652dd21c632090',
        destination_dir='temp',
    ),
    'train-23': download_utils.RemoteFileMetadata(
        filename='acousticbrainz-mediaeval-features-train-23.tar.bz2',
        url='https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-train-23.tar.bz2?download=1',
        checksum='79581967a1be5c52e83be21261d1ef6c',
        destination_dir='temp',
    ),
    'train-45': download_utils.RemoteFileMetadata(
        filename='acousticbrainz-mediaeval-features-train-45.tar.bz2',
        url='https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-train-45.tar.bz2?download=1',
        checksum='0e48fa319fa48e5cf95eea8118d2e882',
        destination_dir='temp',
    ),
    'train-67': download_utils.RemoteFileMetadata(
        filename='acousticbrainz-mediaeval-features-train-67.tar.bz2',
        url='https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-train-67.tar.bz2?download=1',
        checksum='22ca7f1fea8a86459b7fda4530f00070',
        destination_dir='temp',
    ),
    'train-89': download_utils.RemoteFileMetadata(
        filename='acousticbrainz-mediaeval-features-train-89.tar.bz2',
        url='https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-train-89.tar.bz2?download=1',
        checksum='c6e4a2ef1b0e8ed535197b868f8c7302',
        destination_dir='temp',
    ),
    'train-ab': download_utils.RemoteFileMetadata(
        filename='acousticbrainz-mediaeval-features-train-ab.tar.bz2',
        url='https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-train-ab.tar.bz2?download=1',
        checksum='513d5f306dd4f3799c137423ee444051',
        destination_dir='temp',
    ),
    'train-cd': download_utils.RemoteFileMetadata(
        filename='acousticbrainz-mediaeval-features-train-cd.tar.bz2',
        url='https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-train-cd.tar.bz2?download=1',
        checksum='422d75d70d583decec0b2761865092a7',
        destination_dir='temp',
    ),
    'train-ef': download_utils.RemoteFileMetadata(
        filename='acousticbrainz-mediaeval-features-train-ef.tar.bz2',
        url='https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-train-ef.tar.bz2?download=1',
        checksum='021ab25a5fd1b020521824e7fce9c775',
        destination_dir='temp',
    )
}
REMOTE_INDEX = {
    'REMOTE_INDEX': download_utils.RemoteFileMetadata(
        filename='acousticbrainz_genre_index.json.zip',
        url='https://zenodo.org/record/4298580/files/acousticbrainz_genre_index.json.zip?download=1',
        checksum='810f1c003f53cbe58002ba96e6d4d138',
        destination_dir=''
    )
}

DOWNLOAD_INFO = ""


DATA = utils.LargeData('acousticbrainz_genre_index.json', remote_index=REMOTE_INDEX)


class Track(core.Track):
    """AcousticBrainz Genre Dataset track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`

    Attributes:
        track_id (str): track id
    """

    def __init__(self, track_id, data_home, remote_index=None, remote_index_name=None):
        if remote_index is not None and remote_index_name is not None:
            data = utils.LargeData(remote_index_name, remote_index=remote_index)
        else:
            data = DATA

        if track_id not in data.index["tracks"]:
            raise ValueError('{} is not a valid track ID in AcousticBrainz genre Dataset'.format(track_id))

        self.track_id = track_id
        self._data_home = data_home
        self._track_paths = data.index["tracks"][track_id]
        self.path = utils.none_path_join(
            [self._data_home, self._track_paths['data'][0]]
        )

    # Genre
    @property
    def genre(self):
        """Genre: human-labeled genre and subgenres list"""
        return [genre for genre in self.track_id.split('#')[2:]]

    # Music Brainz
    @property
    def mbid(self):
        """mbid: musicbrainz id"""
        return self.track_id.split('#')[0]

    @property
    def mbid_group(self):
        """mbid_group: musicbrainz id group"""
        return self.track_id.split('#')[1]

    # Metadata
    @property
    def artist(self):
        """Artist: metadata artist annotation
        """
        return load_extractor(self.path)["metadata"]["artist"]

    @property
    def title(self):
        """title: metadata title annotation"""
        return load_extractor(self.path)["metadata"]["title"]

    @property
    def date(self):
        """date: metadata date annotation"""
        return load_extractor(self.path)["metadata"]["date"]

    @property
    def file_name(self):
        """File_name: metadata file_name annotation"""
        return load_extractor(self.path)["metadata"]["file_name"]

    @property
    def album(self):
        """Album: metadata album annotation"""
        return load_extractor(self.path)["metadata"]["album"]

    @property
    def tracknumber(self):
        """tracknumber: metadata tracknumber annotation"""
        return load_extractor(self.path)["metadata"]["tracknumber"]

    # Tonal
    @property
    def tonal(self):
        """Tonal: tonal features.
        'tuning_frequency': estimated tuning frequency [Hz]. Algorithms: TuningFrequency
        'tuning_nontempered_energy_ratio' and 'tuning_equal_tempered_deviation'

        'hpcp', 'thpcp': 32-dimensional harmonic pitch class profile (HPCP) and its transposed version. Algorithms: HPCP

        'hpcp_entropy': Shannon entropy of a HPCP vector. Algorithms: Entropy

        'key_key', 'key_scale': Global key feature. Algorithms: Key

        'chords_key', 'chords_scale': Global key extracted from chords detection.

        'chords_strength', 'chords_histogram': : strength of estimated chords and normalized histogram of their
        progression; Algorithms: ChordsDetection, ChordsDescriptors


        'chords_changes_rate', 'chords_number_rate':  chords change rate in the progression; ratio
        of different chords from the total number of chords in the progression; Algorithms: ChordsDetection,
        ChordsDescriptors
        Example:
        ```JSON
        'tonal':{
              'thpcp':[
                 1,
                 0.507318735123,
                 0.0966857150197,
                 0.0426645539701,
                 0.0483965314925,
                 0.179099410772,
                 0.320293188095,
                 0.237049147487,
                 0.2395414114,
                 0.350966274738,
                 0.232010543346,
                 0.0956381931901,
                 0.0849455147982,
                 0.0992767587304,
                 0.220862701535,
                 0.346613913774,
                 0.254877924919,
                 0.144492298365,
                 0.138952225447,
                 0.168216094375,
                 0.390465915203,
                 0.578827261925,
                 0.340706437826,
                 0.0968758687377,
                 0.0831783562899,
                 0.0587093383074,
                 0.0431259125471,
                 0.0452906452119,
                 0.0403949655592,
                 0.0697610229254,
                 0.102411799133,
                 0.0839798152447,
                 0.102026410401,
                 0.12077883631,
                 0.186415702105,
                 0.699734687805
              ],
              'hpcp':{
                 'dmean2':[
                    0.178594380617,
                    0.16023632884,
                    0.103057518601,
                    0.0980740636587,
                    0.110168747604,
                    0.146259948611,
                    0.159311890602,
                    0.152000948787,
                    0.136627301574,
                    0.139202192426,
                    0.171973928809,
                    0.216359615326,
                    0.219214200974,
                    0.197275102139,
                    0.0970009341836,
                    0.070560619235,
                    0.0568987056613,
                    0.0447382815182,
                    0.0319559387863,
                    0.0362393707037,
                    0.0506391376257,
                    0.0647731274366,
                    0.0639156401157,
                    0.0770523697138,
                    0.0981073230505,
                    0.193972751498,
                    0.327600568533,
                    0.227274686098,
                    0.306807488203,
                    0.122404053807,
                    0.0589169487357,
                    0.0582390166819,
                    0.104690216482,
                    0.141766279936,
                    0.151633277535,
                    0.158217191696
                 ],
                 'median':[
                    0.0242512505502,
                    0.0162243917584,
                    0.00442832708359,
                    0.0026245675981,
                    0.00274003460072,
                    0.00746832182631,
                    0.0087266638875,
                    0.00605125585571,
                    0.0032673496753,
                    0.00273101869971,
                    0.00652965530753,
                    0.0492442660034,
                    0.0885675624013,
                    0.0490445457399,
                    0.0065808207728,
                    0.00144797482062,
                    0.00138741335832,
                    0.00163704121951,
                    0.00170974992216,
                    0.00198111264035,
                    0.00269822846167,
                    0.00298195728101,
                    0.00444959057495,
                    0.00482877250761,
                    0.00520649645478,
                    0.0221224334091,
                    0.210975214839,
                    0.357364058495,
                    0.121964357793,
                    0.00403643492609,
                    0.000631547707599,
                    0.00127915048506,
                    0.0120413331315,
                    0.0221591964364,
                    0.0243600532413,
                    0.0220098067075
                 ],
                 'min':[
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                 ],
                 'dvar2':[
                    0.0923194587231,
                    0.0815363973379,
                    0.0511545799673,
                    0.0502935610712,
                    0.0541777983308,
                    0.066973619163,
                    0.0863623097539,
                    0.0762098804116,
                    0.0845957174897,
                    0.0904724001884,
                    0.112700395286,
                    0.102032870054,
                    0.104157529771,
                    0.0877331718802,
                    0.0455346442759,
                    0.0394088961184,
                    0.031592041254,
                    0.0258620958775,
                    0.0166516024619,
                    0.0176277346909,
                    0.0235926751047,
                    0.0331757478416,
                    0.0272852964699,
                    0.034093439579,
                    0.0557818897069,
                    0.094073176384,
                    0.149004563689,
                    0.0981300100684,
                    0.159933894873,
                    0.0621675103903,
                    0.0314343906939,
                    0.0269571710378,
                    0.0393776223063,
                    0.0622402280569,
                    0.0579484254122,
                    0.0726568624377
                 ],
                 'dvar':[
                    0.0379842370749,
                    0.0305933672935,
                    0.0183975361288,
                    0.0187579095364,
                    0.0199990961701,
                    0.0261411704123,
                    0.0364330522716,
                    0.0299046840519,
                    0.0302553288639,
                    0.0323115438223,
                    0.039840798825,
                    0.0383969843388,
                    0.0416979975998,
                    0.0317634791136,
                    0.0161656066775,
                    0.0144214453176,
                    0.010956662707,
                    0.00881491228938,
                    0.00617209728807,
                    0.00639398163185,
                    0.00909593049437,
                    0.0137482509017,
                    0.0105136232451,
                    0.0119437174872,
                    0.0193518865854,
                    0.0306953992695,
                    0.0504642426968,
                    0.0413282848895,
                    0.0514670647681,
                    0.020954452455,
                    0.0108866160735,
                    0.00946747977287,
                    0.0162360239774,
                    0.0270703621209,
                    0.023217510432,
                    0.0273143574595
                 ],
                 'dmean':[
                    0.104321710765,
                    0.094670727849,
                    0.0570936873555,
                    0.0541612058878,
                    0.0615743845701,
                    0.0844450071454,
                    0.0916572734714,
                    0.0889912396669,
                    0.0756299942732,
                    0.0763506665826,
                    0.0960845500231,
                    0.12720541656,
                    0.127985849977,
                    0.115848392248,
                    0.053705625236,
                    0.0392280034721,
                    0.0311841331422,
                    0.0244226101786,
                    0.0177254900336,
                    0.0203565079719,
                    0.0296429470181,
                    0.0375514961779,
                    0.0366800874472,
                    0.0438719019294,
                    0.054726947099,
                    0.105988435447,
                    0.187851026654,
                    0.130715310574,
                    0.172673404217,
                    0.0660371035337,
                    0.0314599238336,
                    0.0316690467298,
                    0.0618434101343,
                    0.0852539539337,
                    0.0900341421366,
                    0.0928411781788
                 ],
                 'max':[
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1
                 ],
                 'var':[
                    0.0845739617944,
                    0.0427777655423,
                    0.0152327995747,
                    0.0151554355398,
                    0.016528274864,
                    0.0412515923381,
                    0.0953552722931,
                    0.0543934851885,
                    0.0322691909969,
                    0.0338524654508,
                    0.0370120033622,
                    0.0703158676624,
                    0.123182617128,
                    0.0527594648302,
                    0.0137857059017,
                    0.0168607961386,
                    0.0096616121009,
                    0.00771625339985,
                    0.0116523122415,
                    0.00652206037194,
                    0.011681237258,
                    0.0242706183344,
                    0.0137251811102,
                    0.0142718972638,
                    0.021222628653,
                    0.0259281843901,
                    0.111432902515,
                    0.183202713728,
                    0.0751374587417,
                    0.0145144108683,
                    0.00686844764277,
                    0.00657867267728,
                    0.029618723318,
                    0.0745138078928,
                    0.0379381924868,
                    0.0437051840127
                 ],
                 'mean':[
                    0.164834395051,
                    0.108965791762,
                    0.04491731897,
                    0.0398954078555,
                    0.0466262027621,
                    0.103730112314,
                    0.16279026866,
                    0.119705654681,
                    0.0678620785475,
                    0.0652601346374,
                    0.0790041685104,
                    0.183385744691,
                    0.271851301193,
                    0.160015776753,
                    0.0454986020923,
                    0.0390654467046,
                    0.0275733564049,
                    0.02025446482,
                    0.0212711505592,
                    0.0189718510956,
                    0.0327638760209,
                    0.0480986014009,
                    0.039441857487,
                    0.0479176007211,
                    0.0567248426378,
                    0.0875517725945,
                    0.328636556864,
                    0.469658792019,
                    0.238266691566,
                    0.0454092957079,
                    0.0200377833098,
                    0.0227298568934,
                    0.0841156095266,
                    0.150428518653,
                    0.111332215369,
                    0.112502731383
                 ]
              },
              'tuning_equal_tempered_deviation':0,
              'chords_changes_rate':0.0403249189258,
              'key_key':'F#',
              'tuning_diatonic_strength':0.670455694199,
              'key_scale':'minor',
              'chords_number_rate':0.00290107331239,
              'tuning_frequency':440.508636475,
              'tuning_nontempered_energy_ratio':0.668684065342,
              'chords_histogram':[
                 52.9445877075,
                 4.64171743393,
                 6.15027570724,
                 0.290107339621,
                 3.04612708092,
                 1.97272992134,
                 0,
                 11.8653898239,
                 0,
                 6.35335063934,
                 3.56832027435,
                 0.0870321989059,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0.290107339621,
                 0.464171737432,
                 0.667246878147,
                 0,
                 0,
                 3.77139544487,
                 3.88743829727
              ],
              'key_strength':0.791583478451,
              'chords_key':'F#',
              'chords_strength':{
                 'dmean2':0.00804366357625,
                 'median':0.547548830509,
                 'min':-1,
                 'dvar2':0.00116975617129,
                 'dvar':0.000607345020398,
                 'dmean':0.00839492119849,
                 'max':0.813877642155,
                 'var':0.0339203439653,
                 'mean':0.524816870689
              },
              'chords_scale':'minor',
              'hpcp_entropy':{
                 'dmean2':1.00705373287,
                 'median':1.56870746613,
                 'min':0,
                 'dvar2':0.648505687714,
                 'dvar':0.2447180897,
                 'dmean':0.585986793041,
                 'max':3.65271377563,
                 'var':0.561654984951,
                 'mean':1.50042426586
              }
           },
        ```
        """
        return load_extractor(self.path)["tonal"]

    # low_level
    @property
    def low_level(self):
        """low_level: low_level track descritors.

        'average_loudness': dynamic range descriptor. It rescales average loudness,
        computed on 2sec windows with 1 sec overlap, into the [0,1] interval. The value of 0 corresponds to signals
        with large dynamic range, 1 corresponds to signal with little dynamic range. Algorithms: Loudness

        'dynamic_complexity': dynamic complexity computed on 2sec windows with 1sec overlap. Algorithms: DynamicComplexity

        'silence_rate_20dB', 'silence_rate_30dB', 'silence_rate_60dB': rate of silent frames in a signal for
        thresholds of 20, 30, and 60 dBs. Algorithms: SilenceRate

        'spectral_rms': spectral RMS. Algorithms: RMS

        'spectral_flux': spectral flux of a signal computed using L2-norm. Algorithms: Flux

        'spectral_centroid', 'spectral_kurtosis', 'spectral_spread', 'spectral_skewness': centroid and central
        moments statistics describing the spectral shape. Algorithms: Centroid, CentralMoments

        'spectral_rolloff': the roll-off frequency of a spectrum. Algorithms: RollOff

        'spectral_decrease': spectral decrease. Algorithms: Decrease

        'hfc': high frequency content descriptor as proposed by Masri. Algorithms: HFC

        'zerocrossingrate' zero-crossing rate. Algorithms: ZeroCrossingRate

        'spectral_energy': spectral energy. Algorithms: Energy

        'spectral_energyband_low', 'spectral_energyband_middle_low', 'spectral_energyband_middle_high',
        'spectral_energyband_high': spectral energy in frequency bands [20Hz, 150Hz], [150Hz, 800Hz], [800Hz, 4kHz],
        and [4kHz, 20kHz]. Algorithms EnergyBand

        'barkbands': spectral energy in 27 Bark bands. Algorithms: BarkBands

        'melbands': spectral energy in 40 mel bands. Algorithms: MFCC

        'erbbands': spectral energy in 40 ERB bands. Algorithms: ERBBands

        'mfcc': the first 13 mel frequency cepstrum coefficients. See algorithm: MFCC

        'gfcc': the first 13 gammatone feature cepstrum coefficients. Algorithms: GFCC

        'barkbands_crest', 'barkbands_flatness_db': crest and flatness computed over energies in Bark bands. Algorithms: Crest, FlatnessDB

        'barkbands_kurtosis', 'barkbands_skewness', 'barkbands_spread': central moments statistics over energies in Bark bands. Algorithms: CentralMoments

        'melbands_crest', 'melbands_flatness_db': crest and flatness computed over energies in mel bands. Algorithms: Crest, FlatnessDB

        'melbands_kurtosis', 'melbands_skewness', 'melbands_spread': central moments statistics over energies in mel bands. Algorithms: CentralMoments

        'erbbands_crest', 'erbbands_flatness_db': crest and flatness computed over energies in ERB bands. Algorithms: Crest, FlatnessDB

        'erbbands_kurtosis', 'erbbands_skewness', 'erbbands_spread': central moments statistics over energies in ERB bands. Algorithms: CentralMoments

        'dissonance': sensory dissonance of a spectrum. Algorithms: Dissonance

        'spectral_entropy': Shannon entropy of a spectrum. Algorithms: Entropy

        'pitch_salience': pitch salience of a spectrum. Algorithms: PitchSalience

        'spectral_complexity': spectral complexity. Algorithms: SpectralComplexity

        'spectral_contrast_coeffs', 'spectral_contrast_valleys': spectral contrast features. Algorithms:
        SpectralContrast

        Example:
        ```JSON
        'lowlevel':{
          'barkbands_spread':{
             'dmean2':3.77554965019,
             'median':4.61414766312,
             'min':0.173020109534,
             'dvar2':61.6504440308,
             'dvar':27.9641170502,
             'dmean':2.44167518616,
             'max':110.2914505,
             'var':124.757987976,
             'mean':7.52866315842
          },
          'melbands_spread':{
             'dmean2':3.78300070763,
             'median':4.92296409607,
             'min':0.410931527615,
             'dvar2':77.338180542,
             'dvar':41.6111984253,
             'dmean':2.51712179184,
             'max':170.462478638,
             'var':301.857910156,
             'mean':8.54182434082
          },
          'hfc':{
             'dmean2':3.50013637543,
             'median':6.10393381119,
             'min':1.06054540254e-16,
             'dvar2':34.0371131897,
             'dvar':14.4582118988,
             'dmean':2.31053972244,
             'max':88.2738571167,
             'var':75.8253250122,
             'mean':8.62131118774
          },
          'barkbands_kurtosis':{
             'dmean2':11.8829393387,
             'median':9.40664482117,
             'min':-1.84666728973,
             'dvar2':625.054504395,
             'dvar':269.846496582,
             'dmean':7.67745256424,
             'max':536.236999512,
             'var':1004.30914307,
             'mean':17.9715843201
          },
          'gfcc':{
             'mean':[
                -132.788909912,
                142.14666748,
                -81.0338973999,
                8.63331604004,
                -58.3307991028,
                -5.39012002945,
                -34.5070724487,
                3.37363839149,
                -21.0019397736,
                -0.66143321991,
                -5.58816480637,
                -14.6182832718,
                1.07953834534
             ],
             'icov':[
                [
                   0.000335385237122,
                   -0.00103988894261,
                   0.000736981979571,
                   -0.00120457413141,
                   0.000940136378631,
                   -0.00114710326307,
                   0.00112895527855,
                   -0.000861310574692,
                   0.000751518004108,
                   -0.000711515196599,
                   0.00037995740422,
                   -0.000630446360447,
                   0.00042732435395
                ],
                [
                   -0.00103988894261,
                   0.00400719698519,
                   -0.00209345272742,
                   0.00360448728316,
                   -0.0037215934135,
                   0.00380552583374,
                   -0.00388551456854,
                   0.00323714734986,
                   -0.00268420949578,
                   0.00262403511442,
                   -0.00163167156279,
                   0.00266133132391,
                   -0.00165268150158
                ],
                [
                   0.000736981979571,
                   -0.00209345272742,
                   0.00341281644069,
                   -0.00294547923841,
                   0.00114159076475,
                   -0.00306878169067,
                   0.00228093308397,
                   -0.00258892471902,
                   0.00158821267541,
                   -0.000897726044059,
                   0.00122907594778,
                   -0.00190949963871,
                   0.000966875872109
                ],
                [
                   -0.00120457413141,
                   0.00360448728316,
                   -0.00294547923841,
                   0.00689155561849,
                   -0.00431259116158,
                   0.00203535147011,
                   -0.00264717591926,
                   0.00288130971603,
                   -0.00135119084734,
                   0.00197674031369,
                   -0.0014328159159,
                   0.00141850474756,
                   -0.000610302726272
                ],
                [
                   0.000940136378631,
                   -0.0037215934135,
                   0.00114159076475,
                   -0.00431259116158,
                   0.00646798405796,
                   -0.00263112364337,
                   0.00132549903356,
                   -0.00174455426168,
                   0.00157945649698,
                   -0.00152322486974,
                   0.000767953693867,
                   -0.00178843981121,
                   0.00122726708651
                ],
                [
                   -0.00114710326307,
                   0.00380552583374,
                   -0.00306878169067,
                   0.00203535147011,
                   -0.00263112364337,
                   0.0112451156601,
                   -0.00653579598293,
                   0.00279242661782,
                   -0.00283282389864,
                   0.00191252457444,
                   -0.00199365057051,
                   0.00332468468696,
                   -0.00153529189993
                ],
                [
                   0.00112895527855,
                   -0.00388551456854,
                   0.00228093308397,
                   -0.00264717591926,
                   0.00132549903356,
                   -0.00653579598293,
                   0.0121742300689,
                   -0.00452526239678,
                   -0.000682854210027,
                   -0.00377777148969,
                   0.00384873338044,
                   -0.00344320596196,
                   0.00246894545853
                ],
                [
                   -0.000861310574692,
                   0.00323714734986,
                   -0.00258892471902,
                   0.00288130971603,
                   -0.00174455426168,
                   0.00279242661782,
                   -0.00452526239678,
                   0.00898661650717,
                   -0.00442420411855,
                   0.00187778519467,
                   -0.00263732252643,
                   0.00327507732436,
                   -0.00180146866478
                ],
                [
                   0.000751518004108,
                   -0.00268420949578,
                   0.00158821267541,
                   -0.00135119084734,
                   0.00157945649698,
                   -0.00283282389864,
                   -0.000682854210027,
                   -0.00442420411855,
                   0.0127948811278,
                   -0.0034028289374,
                   0.000330279202899,
                   -0.00463727954775,
                   0.00367314391769
                ],
                [
                   -0.000711515196599,
                   0.00262403511442,
                   -0.000897726044059,
                   0.00197674031369,
                   -0.00152322486974,
                   0.00191252457444,
                   -0.00377777148969,
                   0.00187778519467,
                   -0.0034028289374,
                   0.0120996944606,
                   -0.00067205511732,
                   -0.000694065121934,
                   -0.000791335711256
                ],
                [
                   0.00037995740422,
                   -0.00163167156279,
                   0.00122907594778,
                   -0.0014328159159,
                   0.000767953693867,
                   -0.00199365057051,
                   0.00384873338044,
                   -0.00263732252643,
                   0.000330279202899,
                   -0.00067205511732,
                   0.0107421576977,
                   -0.00082002527779,
                   -0.00202772649936
                ],
                [
                   -0.000630446360447,
                   0.00266133132391,
                   -0.00190949963871,
                   0.00141850474756,
                   -0.00178843981121,
                   0.00332468468696,
                   -0.00344320596196,
                   0.00327507732436,
                   -0.00463727954775,
                   -0.000694065121934,
                   -0.00082002527779,
                   0.0123253297061,
                   -0.00216704793274
                ],
                [
                   0.00042732435395,
                   -0.00165268150158,
                   0.000966875872109,
                   -0.000610302726272,
                   0.00122726708651,
                   -0.00153529189993,
                   0.00246894545853,
                   -0.00180146866478,
                   0.00367314391769,
                   -0.000791335711256,
                   -0.00202772649936,
                   -0.00216704793274,
                   0.0142371747643
                ]
             ],
             'cov':[
                [
                   37387.5976563,
                   4358.72021484,
                   -1795.53808594,
                   2832.53198242,
                   -316.226989746,
                   564.97265625,
                   -1381.02307129,
                   -544.360961914,
                   -1104.99475098,
                   -119.790603638,
                   434.688079834,
                   -437.863342285,
                   159.637817383
                ],
                [
                   4358.72021484,
                   2446.73657227,
                   122.067451477,
                   205.417373657,
                   738.437866211,
                   7.15356206894,
                   192.322372437,
                   -149.13142395,
                   69.4642028809,
                   -105.095726013,
                   60.6722793579,
                   -85.4167327881,
                   10.4846248627
                ],
                [
                   -1795.53808594,
                   122.067451477,
                   992.309509277,
                   281.873321533,
                   420.629211426,
                   203.193283081,
                   265.133728027,
                   140.490310669,
                   149.350082397,
                   17.3196163177,
                   -55.1355857849,
                   88.7021713257,
                   -61.7136650085
                ],
                [
                   2832.53198242,
                   205.417373657,
                   281.873321533,
                   737.372558594,
                   250.831192017,
                   195.922958374,
                   -12.1163778305,
                   -13.3502483368,
                   -57.6144294739,
                   5.08618783951,
                   14.5830659866,
                   17.5463848114,
                   -28.8936862946
                ],
                [
                   -316.226989746,
                   738.437866211,
                   420.629211426,
                   250.831192017,
                   705.019958496,
                   115.014541626,
                   268.08807373,
                   15.4422664642,
                   135.850067139,
                   -0.211590483785,
                   -22.3415031433,
                   44.0851707458,
                   -47.0426368713
                ],
                [
                   564.97265625,
                   7.15356206894,
                   203.193283081,
                   195.922958374,
                   115.014541626,
                   245.563034058,
                   128.44291687,
                   60.6132316589,
                   59.6934509277,
                   36.5294494629,
                   -13.3313579559,
                   24.6754741669,
                   -31.0790405273
                ],
                [
                   -1381.02307129,
                   192.322372437,
                   265.133728027,
                   -12.1163778305,
                   268.08807373,
                   128.44291687,
                   397.496398926,
                   116.522514343,
                   215.44694519,
                   75.2359313965,
                   -71.2339935303,
                   83.407875061,
                   -67.0485687256
                ],
                [
                   -544.360961914,
                   -149.13142395,
                   140.490310669,
                   -13.3502483368,
                   15.4422664642,
                   60.6132316589,
                   116.522514343,
                   232.467895508,
                   101.961425781,
                   34.4596824646,
                   1.58056282997,
                   21.653137207,
                   -17.5421600342
                ],
                [
                   -1104.99475098,
                   69.4642028809,
                   149.350082397,
                   -57.6144294739,
                   135.850067139,
                   59.6934509277,
                   215.44694519,
                   101.961425781,
                   237.031005859,
                   65.3951339722,
                   -33.0408248901,
                   76.529510498,
                   -51.6921310425
                ],
                [
                   -119.790603638,
                   -105.095726013,
                   17.3196163177,
                   5.08618783951,
                   -0.211590483785,
                   36.5294494629,
                   75.2359313965,
                   34.4596824646,
                   65.3951339722,
                   130.037628174,
                   -18.586517334,
                   47.9396896362,
                   -19.2859573364
                ],
                [
                   434.688079834,
                   60.6722793579,
                   -55.1355857849,
                   14.5830659866,
                   -22.3415031433,
                   -13.3313579559,
                   -71.2339935303,
                   1.58056282997,
                   -33.0408248901,
                   -18.586517334,
                   124.860694885,
                   -20.3170909882,
                   33.5888900757
                ],
                [
                   -437.863342285,
                   -85.4167327881,
                   88.7021713257,
                   17.5463848114,
                   44.0851707458,
                   24.6754741669,
                   83.407875061,
                   21.653137207,
                   76.529510498,
                   47.9396896362,
                   -20.3170909882,
                   133.77961731,
                   -14.5192298889
                ],
                [
                   159.637817383,
                   10.4846248627,
                   -61.7136650085,
                   -28.8936862946,
                   -47.0426368713,
                   -31.0790405273,
                   -67.0485687256,
                   -17.5421600342,
                   -51.6921310425,
                   -19.2859573364,
                   33.5888900757,
                   -14.5192298889,
                   94.56640625
                ]
             ]
          },
          'spectral_energyband_middle_low':{
             'dmean2':0.00843009725213,
             'median':0.0115021476522,
             'min':1.91154490012e-22,
             'dvar2':0.000183915923117,
             'dvar':7.44249846321e-05,
             'dmean':0.00546479970217,
             'max':0.159320950508,
             'var':0.000311762880301,
             'mean':0.0168608538806
          },
          'melbands_crest':{
             'dmean2':4.65500926971,
             'median':16.3166275024,
             'min':1.69419360161,
             'dvar2':17.4051208496,
             'dvar':6.91288948059,
             'dmean':2.90407824516,
             'max':34.4925079346,
             'var':34.4553756714,
             'mean':16.8213863373
          },
          'spectral_kurtosis':{
             'dmean2':9.29079723358,
             'median':11.0774097443,
             'min':-1.27365171909,
             'dvar2':132.717971802,
             'dvar':51.2053985596,
             'dmean':5.68827199936,
             'max':104.172729492,
             'var':182.843307495,
             'mean':15.3059854507
          },
          'spectral_rms':{
             'dmean2':0.00111874949653,
             'median':0.00479687564075,
             'min':3.10516239457e-12,
             'dvar2':2.06636605071e-06,
             'dvar':8.36767185319e-07,
             'dmean':0.000726740399841,
             'max':0.0170408394188,
             'var':8.51508048072e-06,
             'mean':0.00489136390388
          },
          'zerocrossingrate':{
             'dmean2':0.00668838992715,
             'median':0.0185546875,
             'min':0.0029296875,
             'dvar2':0.000136891874718,
             'dvar':0.000114278678666,
             'dmean':0.00550990412012,
             'max':0.5849609375,
             'var':0.00505612557754,
             'mean':0.0347590073943
          },
          'silence_rate_60dB':{
             'dmean2':0.031930334866,
             'median':0,
             'min':0,
             'dvar2':0.0338160544634,
             'dvar':0.0157057177275,
             'dmean':0.0159628503025,
             'max':1,
             'var':0.122698590159,
             'mean':0.143209517002
          },
          'erbbands_kurtosis':{
             'dmean2':4.79220151901,
             'median':3.30026721954,
             'min':-1.8656873703,
             'dvar2':92.9352035522,
             'dvar':38.6438560486,
             'dmean':3.11310648918,
             'max':196.060791016,
             'var':160.621276855,
             'mean':6.7561750412
          },
          'erbbands':{
             'dmean2':[
                0.155370026827,
                1.9540656805,
                6.09391450882,
                6.8454875946,
                13.5909948349,
                20.6750030518,
                19.9079971313,
                13.2817964554,
                25.3594303131,
                12.8928041458,
                15.4322433472,
                9.52532577515,
                12.7656536102,
                6.12102746964,
                3.23051857948,
                2.48092389107,
                2.07177829742,
                3.00173521042,
                3.23087072372,
                1.25146484375,
                1.35807836056,
                1.22470033169,
                1.13416159153,
                1.25142753124,
                1.29189109802,
                1.76435148716,
                2.19070768356,
                1.46817576885,
                1.3724039793,
                0.810075759888,
                0.544243037701,
                0.591998457909,
                0.291299253702,
                0.133352503181,
                0.0749251097441,
                0.0313248969615,
                0.00559062883258,
                0.000832212448586,
                8.8281056378e-05,
                1.94007607206e-05
             ],
             'median':[
                0.0470806881785,
                0.798209190369,
                1.99921917915,
                2.62412786484,
                6.44044494629,
                8.76387023926,
                10.656835556,
                9.23368263245,
                14.6313858032,
                7.24559736252,
                5.80149459839,
                3.74936962128,
                5.79891872406,
                3.25933170319,
                1.83694386482,
                1.30305314064,
                1.20966291428,
                1.2380900383,
                1.75359368324,
                0.764325618744,
                0.488044530153,
                0.382896959782,
                0.322072476149,
                0.258826166391,
                0.261801838875,
                0.469595551491,
                0.333578705788,
                0.0666380524635,
                0.0169108975679,
                0.0124986991286,
                0.0473246648908,
                0.055234670639,
                0.0339137017727,
                0.0146979326382,
                0.00507948361337,
                0.00162576022558,
                0.000380026351195,
                9.82415149338e-05,
                3.17100420943e-05,
                2.91496562568e-05
             ],
             'min':[
                1.16677993111e-22,
                3.50039495047e-22,
                1.1153717718e-21,
                5.34195582188e-21,
                1.04094209663e-20,
                2.88563119912e-20,
                6.71323629277e-20,
                6.45778740997e-20,
                1.61343633505e-19,
                2.26918622135e-19,
                3.09646532009e-19,
                3.13455414387e-19,
                4.1692919497e-19,
                5.19374505708e-19,
                8.63080044343e-19,
                1.02024498111e-18,
                1.81746200995e-18,
                2.33893796801e-18,
                2.58153535921e-18,
                3.88429581353e-18,
                4.22434852258e-18,
                4.79898469027e-18,
                5.8320327729e-18,
                5.42632218914e-18,
                6.69074551041e-18,
                9.29391102225e-18,
                8.91652806575e-18,
                1.06487551106e-17,
                1.00586221278e-17,
                1.21652344266e-17,
                1.27961871184e-17,
                1.04252277481e-17,
                1.0594467247e-17,
                8.08031960399e-18,
                6.86025715179e-18,
                3.92254381788e-18,
                2.35179938576e-18,
                9.61571199108e-19,
                2.52459219415e-19,
                2.66188320551e-20
             ],
             'dvar2':[
                0.171826034784,
                26.6956825256,
                162.282745361,
                293.938659668,
                671.892822266,
                2503.62158203,
                1745.20410156,
                1593.59387207,
                2598.02783203,
                781.077148438,
                1939.0859375,
                792.426574707,
                1627.14001465,
                376.117004395,
                77.2415084839,
                34.1384353638,
                19.2012786865,
                84.0574111938,
                182.903686523,
                8.93897819519,
                17.6475200653,
                21.2283821106,
                13.1880941391,
                32.9881019592,
                40.0318031311,
                24.182138443,
                77.1399383545,
                46.9635429382,
                60.0843887329,
                19.9275169373,
                8.4421005249,
                6.51240158081,
                1.50587880611,
                0.386726230383,
                0.125124797225,
                0.0270316954702,
                0.000720594485756,
                1.38129771585e-05,
                1.26409872792e-07,
                1.13540565838e-09
             ],
             'dvar':[
                0.0638966411352,
                9.04620361328,
                60.4135551453,
                106.592819214,
                247.684951782,
                950.867370605,
                774.599060059,
                828.04296875,
                1081.49926758,
                343.837585449,
                877.836608887,
                387.39944458,
                882.905578613,
                236.651809692,
                38.9343681335,
                13.5093927383,
                7.27470493317,
                34.5314559937,
                64.0410003662,
                3.73893380165,
                6.23876857758,
                7.46507692337,
                4.74067544937,
                11.8515691757,
                15.5274372101,
                10.4364891052,
                30.7011890411,
                20.3718738556,
                27.5566539764,
                8.41174507141,
                3.59223079681,
                2.73822712898,
                0.665752291679,
                0.186611428857,
                0.0575682520866,
                0.0109113147482,
                0.000317029160215,
                6.10745246377e-06,
                5.13876265984e-08,
                4.44208586581e-10
             ],
             'dmean':[
                0.0976967662573,
                1.20870876312,
                3.61620092392,
                4.04383277893,
                8.24364376068,
                13.2294816971,
                13.0249862671,
                9.45654392242,
                16.0714702606,
                8.72541618347,
                10.4990024567,
                6.58109045029,
                9.32448387146,
                4.47474527359,
                2.16594624519,
                1.5720859766,
                1.27258121967,
                1.87077200413,
                1.99565172195,
                0.775347948074,
                0.807783842087,
                0.721225202084,
                0.67847007513,
                0.736807763577,
                0.787335336208,
                1.1302844286,
                1.35330307484,
                0.919236600399,
                0.889913380146,
                0.500978410244,
                0.338999509811,
                0.36962673068,
                0.185711205006,
                0.0888766944408,
                0.0485399067402,
                0.0193678438663,
                0.00350073818117,
                0.000519978057127,
                5.45099173905e-05,
                1.2333954146e-05
             ],
             'max':[
                4.59040546417,
                48.3324165344,
                186.147613525,
                181.793228149,
                285.386932373,
                714.273376465,
                804.361633301,
                1157.68188477,
                784.149780273,
                393.409667969,
                1280.98937988,
                663.18963623,
                1095.42687988,
                741.743835449,
                361.424804688,
                96.5252685547,
                67.2129440308,
                252.501235962,
                475.873474121,
                61.139125824,
                45.9514122009,
                98.1905441284,
                62.9890213013,
                108.605262756,
                148.698745728,
                76.2919235229,
                150.187713623,
                114.430778503,
                195.404312134,
                90.1577682495,
                100.268875122,
                65.9967575073,
                28.8285102844,
                12.7337093353,
                7.53277730942,
                2.90803432465,
                0.54224050045,
                0.06959874928,
                0.00760272005573,
                0.000336901168339
             ],
             'var':[
                0.133741334081,
                18.2778759003,
                162.158370972,
                153.360900879,
                478.75793457,
                2895.30761719,
                2804.58911133,
                3883.76977539,
                3094.73168945,
                1370.14294434,
                3825.02514648,
                1163.2557373,
                2984.50805664,
                742.161193848,
                127.385681152,
                32.1669616699,
                15.2594537735,
                62.9382247925,
                95.4504013062,
                8.84553527832,
                7.59517049789,
                7.79293775558,
                6.2437338829,
                12.2372579575,
                17.7772121429,
                20.929901123,
                38.6162414551,
                29.8375015259,
                47.6405296326,
                10.4680538177,
                4.30644655228,
                3.74839806557,
                1.00600230694,
                0.290444642305,
                0.079219520092,
                0.0108919711784,
                0.00037447817158,
                7.48626325731e-06,
                5.80387968796e-08,
                1.77149162006e-09
             ],
             'mean':[
                0.17717859149,
                2.47696065903,
                7.02564954758,
                6.21073246002,
                14.1452331543,
                29.5357017517,
                29.6157798767,
                26.2096424103,
                33.1595687866,
                20.6438293457,
                22.7253684998,
                12.7716827393,
                19.1616344452,
                9.48248100281,
                4.40138339996,
                3.10452365875,
                2.535810709,
                3.39984416962,
                3.69348263741,
                1.59364187717,
                1.3808709383,
                1.16064369678,
                1.18491280079,
                1.10196530819,
                1.11243593693,
                2.0187587738,
                1.90224671364,
                1.06476140022,
                1.07502770424,
                0.519663453102,
                0.423738330603,
                0.558644175529,
                0.278624206781,
                0.127757504582,
                0.0597115457058,
                0.0200475491583,
                0.00401964969933,
                0.000672693189699,
                8.74161269167e-05,
                4.06568178732e-05
             ]
          },
          'spectral_strongpeak':{
             'dmean2':0.471793323755,
             'median':0.580037772655,
             'min':0,
             'dvar2':0.43769544363,
             'dvar':0.181167602539,
             'dmean':0.283494114876,
             'max':7.49854946136,
             'var':0.463859498501,
             'mean':0.76316267252
          },
          'spectral_energy':{
             'dmean2':0.0156581457704,
             'median':0.023585267365,
             'min':9.88308448854e-21,
             'dvar2':0.000749870028812,
             'dvar':0.000286899041384,
             'dmean':0.00993872154504,
             'max':0.297649979591,
             'var':0.00124526151922,
             'mean':0.0332516431808
          },
          'average_loudness':0.396704643965,
          'spectral_rolloff':{
             'dmean2':314.9269104,
             'median':409.130859375,
             'min':86.1328125,
             'dvar2':1656800.625,
             'dvar':663627.9375,
             'dmean':179.357223511,
             'max':20865.6738281,
             'var':7784192.5,
             'mean':934.389343262
          },
          'spectral_centroid':{
             'dmean2':167.721115112,
             'median':406.963592529,
             'min':120.362541199,
             'dvar2':183124.171875,
             'dvar':97172.4140625,
             'dmean':111.819389343,
             'max':11666.9931641,
             'var':1509851.125,
             'mean':647.602722168
          },
          'pitch_salience':{
             'dmean2':0.115531079471,
             'median':0.515319347382,
             'min':0.0617939718068,
             'dvar2':0.010652013123,
             'dvar':0.00418812222779,
             'dmean':0.0718740522861,
             'max':0.992550611496,
             'var':0.0196558814496,
             'mean':0.500831186771
          },
          'spectral_energyband_middle_high':{
             'dmean2':0.000458390597487,
             'median':0.000449149287306,
             'min':1.18205795803e-21,
             'dvar2':1.78842594778e-06,
             'dvar':1.09112158952e-06,
             'dmean':0.000331070652464,
             'max':0.0536588542163,
             'var':4.30856334788e-06,
             'mean':0.000908457208425
          },
          'spectral_contrast_coeffs':{
             'dmean2':[
                0.075445048511,
                0.0692735165358,
                0.0563321448863,
                0.0394218601286,
                0.0291569642723,
                0.0262591484934
             ],
             'median':[
                -0.60754275322,
                -0.685104012489,
                -0.73859077692,
                -0.805804908276,
                -0.830693006516,
                -0.807184457779
             ],
             'min':[
                -0.999856352806,
                -0.999473273754,
                -0.998966515064,
                -0.998527228832,
                -0.990249931812,
                -0.974876463413
             ],
             'dvar2':[
                0.00411217473447,
                0.00299453851767,
                0.00199586083181,
                0.000966619641986,
                0.000605013628956,
                0.000610490329564
             ],
             'dvar':[
                0.0016761609586,
                0.00118540902622,
                0.000848070310894,
                0.000366522319382,
                0.000253112521023,
                0.000263271882432
             ],
             'dmean':[
                0.0463223233819,
                0.0427533686161,
                0.0365649424493,
                0.0242002625018,
                0.0179802495986,
                0.0165389757603
             ],
             'max':[
                -0.226020470262,
                -0.421985566616,
                -0.456726998091,
                -0.603531241417,
                -0.644345462322,
                -0.664681434631
             ],
             'var':[
                0.0148717714474,
                0.00682240398601,
                0.00509816454723,
                0.00238920329139,
                0.00175602198578,
                0.00189632317051
             ],
             'mean':[
                -0.599549412727,
                -0.686651110649,
                -0.737253189087,
                -0.80643594265,
                -0.830224752426,
                -0.810669541359
             ]
          },
          'melbands_skewness':{
             'dmean2':1.17803037167,
             'median':2.42587566376,
             'min':-3.52072119713,
             'dvar2':1.50288712978,
             'dvar':0.662373363972,
             'dmean':0.75628978014,
             'max':17.1603431702,
             'var':4.27477741241,
             'mean':2.79396009445
          },
          'spectral_spread':{
             'dmean2':841852.625,
             'median':3183811.5,
             'min':796449.125,
             'dvar2':1372237070340.0,
             'dvar':770980315136,
             'dmean':550441,
             'max':42968820,
             'var':20359843676200.0,
             'mean':3944053.75
          },
          'dissonance':{
             'dmean2':0.0543997623026,
             'median':0.418845593929,
             'min':0,
             'dvar2':0.00281463400461,
             'dvar':0.00102427392267,
             'dmean':0.0327908322215,
             'max':0.499892026186,
             'var':0.00340241170488,
             'mean':0.407838582993
          },
          'spectral_skewness':{
             'dmean2':0.774191617966,
             'median':2.71820521355,
             'min':-0.0772252082825,
             'dvar2':0.648099601269,
             'dvar':0.268252104521,
             'dmean':0.491852223873,
             'max':8.83276748657,
             'var':1.44770753384,
             'mean':2.8792078495
          },
          'spectral_flux':{
             'dmean2':0.0317656695843,
             'median':0.0563769564033,
             'min':6.41477843066e-11,
             'dvar2':0.00152398855425,
             'dvar':0.000613985524978,
             'dmean':0.0196313727647,
             'max':0.376720368862,
             'var':0.00241715926677,
             'mean':0.0636893138289
          },
          'spectral_contrast_valleys':{
             'dmean2':[
                0.646871805191,
                0.542208790779,
                0.462133824825,
                0.402661472559,
                0.373252421618,
                0.432622075081
             ],
             'median':[
                -6.90146636963,
                -6.95647907257,
                -8.19164657593,
                -8.81689071655,
                -8.85592269897,
                -11.7539710999
             ],
             'min':[
                -28.2046070099,
                -28.167388916,
                -27.7276287079,
                -27.683757782,
                -27.5231952667,
                -27.3732585907
             ],
             'dvar2':[
                0.335758417845,
                0.256331473589,
                0.213166058064,
                0.193465054035,
                0.258489072323,
                0.363954037428
             ],
             'dvar':[
                0.143520131707,
                0.123868964612,
                0.108743295074,
                0.102052271366,
                0.135346651077,
                0.190870583057
             ],
             'dmean':[
                0.386126875877,
                0.33280557394,
                0.289705336094,
                0.253437131643,
                0.251051306725,
                0.305164188147
             ],
             'max':[
                -4.79679965973,
                -4.98788738251,
                -6.05839586258,
                -6.38532352448,
                -6.09020709991,
                -7.89750099182
             ],
             'var':[
                9.35335063934,
                7.33776664734,
                6.7517580986,
                6.53927278519,
                6.4737868309,
                5.02813100815
             ],
             'mean':[
                -8.01363182068,
                -7.50632286072,
                -8.68572902679,
                -9.32727813721,
                -9.35889053345,
                -12.083521843
             ]
          },
          'erbbands_flatness_db':{
             'dmean2':0.0316766388714,
             'median':0.226177275181,
             'min':0.0269191432744,
             'dvar2':0.00102363585029,
             'dvar':0.000487880315632,
             'dmean':0.0213352981955,
             'max':0.396401196718,
             'var':0.00340407574549,
             'mean':0.224541649222
          },
          'spectral_energyband_high':{
             'dmean2':0.000141146418173,
             'median':2.087388566e-05,
             'min':6.87951817514e-21,
             'dvar2':3.10927930514e-07,
             'dvar':1.3840181623e-07,
             'dmean':9.1254900326e-05,
             'max':0.0088831121102,
             'var':2.30491593811e-07,
             'mean':0.000139512223541
          },
          'spectral_entropy':{
             'dmean2':0.34174773097,
             'median':6.4349899292,
             'min':4.49743747711,
             'dvar2':0.141199737787,
             'dvar':0.0708768591285,
             'dmean':0.230730444193,
             'max':9.81969070435,
             'var':0.559852838516,
             'mean':6.51995754242
          },
          'silence_rate_20dB':{
             'dmean2':0,
             'median':1,
             'min':1,
             'dvar2':0,
             'dvar':0,
             'dmean':0,
             'max':1,
             'var':0,
             'mean':1
          },
          'melbands_flatness_db':{
             'dmean2':0.0418821498752,
             'median':0.354551494122,
             'min':0.00273722736165,
             'dvar2':0.00189169729128,
             'dvar':0.000936710159294,
             'dmean':0.0284409653395,
             'max':0.661066353321,
             'var':0.0087356120348,
             'mean':0.347749531269
          },
          'barkbands':{
             'dmean2':[
                0.000251785037108,
                0.00744329812005,
                0.00573536194861,
                0.00261709629558,
                0.00558962812647,
                0.00155689241365,
                0.0015633541625,
                0.000814746192191,
                0.000519699417055,
                0.000325592875015,
                7.4161333032e-05,
                7.76825254434e-05,
                4.86378448841e-05,
                7.80276459409e-05,
                3.72168069589e-05,
                2.48712713073e-05,
                2.18261175178e-05,
                2.19392659346e-05,
                2.50503071584e-05,
                3.70183952327e-05,
                4.66152014269e-05,
                3.82112302759e-05,
                2.01323528017e-05,
                2.17371089093e-05,
                1.09307093226e-05,
                4.52189078715e-06,
                5.29523674686e-07
             ],
             'median':[
                5.92242067796e-05,
                0.00329472497106,
                0.00157038471662,
                0.000795384054072,
                0.00366233312525,
                0.000976994633675,
                0.000784343632404,
                0.000253827951383,
                0.000182630639756,
                0.000121456192574,
                2.20406764129e-05,
                3.4340446291e-05,
                2.49191616604e-05,
                2.89313320536e-05,
                2.08984147321e-05,
                8.40990105644e-06,
                6.00316798227e-06,
                5.43660507901e-06,
                4.37595781477e-06,
                1.01593032014e-05,
                3.94767266698e-06,
                3.66468839275e-07,
                1.51433368956e-06,
                2.49954291576e-06,
                9.89530235529e-07,
                2.6774208095e-07,
                1.01745420977e-07
             ],
             'min':[
                7.69831916184e-26,
                1.43852888144e-24,
                6.82100562524e-25,
                1.65856703702e-24,
                5.76648644247e-24,
                7.52767205591e-24,
                7.67233494643e-24,
                7.28568581779e-24,
                1.18425802798e-23,
                8.62740095578e-24,
                1.24652905124e-23,
                1.67256766148e-23,
                2.36700791169e-23,
                3.65101329906e-23,
                4.04837168781e-23,
                6.11761442672e-23,
                5.89867838871e-23,
                1.0407067687e-22,
                1.19281764212e-22,
                1.43823933793e-22,
                2.11526387438e-22,
                3.6144331878e-22,
                4.12750655351e-22,
                6.1340309377e-22,
                8.73483743707e-22,
                1.28205926752e-21,
                1.95624458629e-21
             ],
             'dvar2':[
                3.21653544688e-07,
                0.000401406636229,
                0.000163253615028,
                3.34817450494e-05,
                0.000120087439427,
                1.62084816111e-05,
                9.27351266e-06,
                5.30270654053e-06,
                2.38700590671e-06,
                1.48546246237e-06,
                8.03174060593e-08,
                2.74359894803e-08,
                1.07636939362e-08,
                1.09485931432e-07,
                5.52310330804e-09,
                5.0829811471e-09,
                6.37944141957e-09,
                4.344539839e-09,
                1.31562210015e-08,
                1.07528972393e-08,
                4.03833375628e-08,
                4.22653130272e-08,
                1.08696731616e-08,
                7.74770025913e-09,
                2.4526389808e-09,
                5.22436371941e-10,
                4.29029034521e-12
             ],
             'dvar':[
                1.09921046487e-07,
                0.000138842180604,
                6.26410765108e-05,
                1.19018168334e-05,
                4.64117874799e-05,
                8.05981926533e-06,
                3.78904314857e-06,
                2.36715004576e-06,
                1.13883163522e-06,
                9.54234451456e-07,
                4.1806835327e-08,
                1.14687921382e-08,
                4.12756895329e-09,
                3.95236448014e-08,
                2.31352959013e-09,
                1.8555786907e-09,
                2.25845475654e-09,
                1.55641366462e-09,
                5.03912112038e-09,
                4.61928717499e-09,
                1.63813105303e-08,
                1.9279980279e-08,
                4.56319160236e-09,
                3.35304006782e-09,
                1.18862097942e-09,
                2.16466178315e-10,
                1.77529052656e-12
             ],
             'dmean':[
                0.000142908320413,
                0.00463288649917,
                0.00339626800269,
                0.00152234872803,
                0.00357294804417,
                0.00106503965799,
                0.000990229425952,
                0.00055689929286,
                0.000363065279089,
                0.000241209723754,
                4.96479296999e-05,
                4.96687462146e-05,
                2.96267771773e-05,
                4.83272233396e-05,
                2.32252987189e-05,
                1.48934523168e-05,
                1.28821375256e-05,
                1.30712951432e-05,
                1.48638500832e-05,
                2.34611979977e-05,
                2.85244495899e-05,
                2.46277140832e-05,
                1.23176141642e-05,
                1.38068808155e-05,
                7.20648176866e-06,
                2.80133508568e-06,
                3.24918488559e-07
             ],
             'max':[
                0.0113403685391,
                0.18680870533,
                0.192078128457,
                0.0564892068505,
                0.149104237556,
                0.0980352386832,
                0.0445082113147,
                0.0572102479637,
                0.0312145259231,
                0.048555765301,
                0.0124060390517,
                0.00252745556645,
                0.00156483612955,
                0.0110217733309,
                0.00176047172863,
                0.000853136763908,
                0.0015601683408,
                0.000901860010345,
                0.00223719887435,
                0.00174049171619,
                0.00260840915143,
                0.00502468971536,
                0.00343828741461,
                0.00164251134265,
                0.00100717623718,
                0.000411604705732,
                3.56238015229e-05
             ],
             'var':[
                1.16844688591e-07,
                0.000285816495307,
                0.000145747937495,
                1.53430373757e-05,
                0.000139252311783,
                3.58041470463e-05,
                1.03137081169e-05,
                1.06608285932e-05,
                3.50042751052e-06,
                3.04003583551e-06,
                1.56786285288e-07,
                3.1402521472e-08,
                7.94575427676e-09,
                5.93943951799e-08,
                5.60282087392e-09,
                2.93664736972e-09,
                2.46096965029e-09,
                2.24578422525e-09,
                5.96258864505e-09,
                8.92516638373e-09,
                2.00907237513e-08,
                3.25885984864e-08,
                5.23340926151e-09,
                5.02693575655e-09,
                1.80405057559e-09,
                2.27331597991e-10,
                2.10272771417e-12
             ],
             'mean':[
                0.000180062852451,
                0.00956722442061,
                0.00562690477818,
                0.00209049577825,
                0.0081444773823,
                0.00276044150814,
                0.0019435180584,
                0.00118357490283,
                0.000673091097269,
                0.000479065522086,
                9.14859338081e-05,
                9.18451914913e-05,
                5.43506284885e-05,
                8.00620691734e-05,
                4.41121446784e-05,
                2.52980298683e-05,
                1.98978286789e-05,
                2.19346657104e-05,
                2.04777825275e-05,
                4.1348484956e-05,
                3.44802683685e-05,
                2.86712220259e-05,
                1.45752865137e-05,
                2.11839760595e-05,
                9.68729091255e-06,
                2.99706630358e-06,
                4.51435681725e-07
             ]
          },
          'erbbands_skewness':{
             'dmean2':0.703747689724,
             'median':1.44177865982,
             'min':-4.42754793167,
             'dvar2':0.519847810268,
             'dvar':0.240549817681,
             'dmean':0.472198069096,
             'max':8.68959999084,
             'var':1.67088627815,
             'mean':1.61603832245
          },
          'erbbands_spread':{
             'dmean2':9.44425773621,
             'median':16.8647537231,
             'min':0.735369086266,
             'dvar2':217.975357056,
             'dvar':99.0579681396,
             'dmean':6.09834051132,
             'max':129.042327881,
             'var':291.183502197,
             'mean':21.5323677063
          },
          'melbands_kurtosis':{
             'dmean2':15.8210096359,
             'median':13.5133676529,
             'min':-1.80261170864,
             'dvar2':699.797851563,
             'dvar':312.316711426,
             'dmean':10.0475234985,
             'max':438.851470947,
             'var':1335.67016602,
             'mean':23.9767551422
          },
          'melbands':{
             'dmean2':[
                0.00172271393239,
                0.0028639791999,
                0.00130092119798,
                0.00119744555559,
                0.000777206441853,
                0.000289778137812,
                0.00036306885886,
                0.000153109373059,
                0.000109968226752,
                6.65166007821e-05,
                6.73365502735e-05,
                3.01256568491e-05,
                1.07165433292e-05,
                9.62084595812e-06,
                5.95519986746e-06,
                5.0285530051e-06,
                5.94258654019e-06,
                6.32667797618e-06,
                2.30518594435e-06,
                1.63067272752e-06,
                1.67402868101e-06,
                1.37042968618e-06,
                1.07786047465e-06,
                1.14442002541e-06,
                1.0724271533e-06,
                9.32206603466e-07,
                1.06137486e-06,
                1.30983482904e-06,
                1.45154660913e-06,
                1.22177641515e-06,
                8.34700983887e-07,
                8.37400136788e-07,
                7.07163508196e-07,
                3.70337687627e-07,
                2.67761379291e-07,
                4.12409548289e-07,
                3.85330025665e-07,
                2.20581213739e-07,
                1.52565675648e-07,
                1.07366794566e-07
             ],
             'median':[
                0.000734908506274,
                0.00121889752336,
                0.000665382540319,
                0.000621218991,
                0.000411149463616,
                0.000205222109798,
                0.000218954199227,
                6.79566437611e-05,
                3.9692127757e-05,
                2.26042830036e-05,
                2.91399228445e-05,
                1.30708385768e-05,
                4.35300262325e-06,
                4.36134632764e-06,
                2.19200137508e-06,
                2.58786667473e-06,
                2.15917543755e-06,
                3.00149054056e-06,
                1.32656145979e-06,
                6.28741588571e-07,
                5.01944782627e-07,
                3.60132162314e-07,
                3.15562544984e-07,
                2.54243559539e-07,
                1.90184749727e-07,
                1.57891008712e-07,
                1.98550267783e-07,
                3.32625972987e-07,
                2.51221081271e-07,
                6.43234230324e-08,
                1.97166425409e-08,
                8.54544879303e-09,
                4.19046619626e-09,
                5.26191534789e-09,
                1.61596958037e-08,
                4.36177458596e-08,
                2.63575703485e-08,
                2.01333527627e-08,
                1.89701943043e-08,
                8.24113222109e-09
             ],
             'min':[
                1.04209097504e-24,
                8.50679299805e-25,
                1.20739303064e-24,
                1.34589905901e-24,
                2.11273516497e-24,
                1.46885180717e-24,
                2.25362572264e-24,
                2.7823938626e-24,
                3.12258638089e-24,
                1.8157373172e-24,
                1.93895896831e-24,
                1.47978689843e-24,
                2.37347183795e-24,
                2.0434603585e-24,
                3.5313666078e-24,
                3.81188160139e-24,
                4.20812722294e-24,
                4.13587781922e-24,
                2.86225991558e-24,
                3.59861502781e-24,
                4.07103700268e-24,
                3.91057204494e-24,
                5.53456409734e-24,
                2.99525377838e-24,
                3.63237156924e-24,
                3.17652020933e-24,
                4.11362839181e-24,
                5.41004910877e-24,
                5.02287420434e-24,
                4.14257879818e-24,
                4.88326436235e-24,
                5.06936887723e-24,
                5.88511968413e-24,
                6.42847682045e-24,
                6.27517156428e-24,
                5.31778709274e-24,
                5.5506268831e-24,
                6.65779040256e-24,
                5.69794271284e-24,
                6.98895342171e-24
             ],
             'dvar2':[
                2.16548287426e-05,
                3.86730644095e-05,
                7.61996307119e-06,
                6.49060848446e-06,
                2.52592190009e-06,
                6.99362772139e-07,
                5.13925840551e-07,
                1.31413543158e-07,
                8.9306958273e-08,
                3.88946226337e-08,
                4.58619844323e-08,
                1.08029931667e-08,
                1.50165835322e-09,
                3.97367277971e-10,
                2.12252992959e-10,
                1.47212159129e-10,
                3.39505368263e-10,
                9.21646148289e-10,
                2.89883742011e-11,
                2.13446534753e-11,
                2.96355544271e-11,
                2.3695788387e-11,
                1.79878421186e-11,
                1.13467525306e-11,
                2.87529600823e-11,
                1.8828966164e-11,
                1.9194123721e-11,
                1.26106382281e-11,
                2.72763756026e-11,
                4.28804422525e-11,
                1.61675343252e-11,
                2.84309121384e-11,
                1.39354300668e-11,
                4.58396211614e-12,
                2.22459377308e-12,
                3.3354731737e-12,
                2.94702521934e-12,
                8.93604905784e-13,
                3.73887607073e-13,
                2.76691755445e-13
             ],
             'dvar':[
                7.8725252024e-06,
                1.36070966619e-05,
                2.82730775325e-06,
                2.42242253989e-06,
                1.10811868126e-06,
                3.59321433052e-07,
                2.10907799669e-07,
                5.75801237801e-08,
                4.06501747818e-08,
                1.90578415271e-08,
                2.54665692978e-08,
                6.9111956158e-09,
                7.78625830478e-10,
                1.74797468167e-10,
                7.78754699615e-11,
                6.10350381347e-11,
                1.35336339357e-10,
                3.20627108197e-10,
                1.17505493183e-11,
                8.31899826803e-12,
                1.04459392525e-11,
                8.35705029484e-12,
                6.54408775433e-12,
                4.09306095961e-12,
                1.02345987588e-11,
                7.32745027848e-12,
                7.78357326509e-12,
                5.32659940461e-12,
                1.13549759873e-11,
                1.63233124933e-11,
                7.36452218653e-12,
                1.26036282105e-11,
                6.08182731607e-12,
                1.84354484803e-12,
                9.08526671863e-13,
                1.50010873948e-12,
                1.16438564519e-12,
                3.87512342093e-13,
                1.746762999e-13,
                1.31208913634e-13
             ],
             'dmean':[
                0.00108983309474,
                0.00171902857255,
                0.000780004891567,
                0.000747257261537,
                0.000511231017299,
                0.000205747084692,
                0.000230121877394,
                0.000104961669422,
                7.44314384065e-05,
                4.65425073344e-05,
                4.9290170864e-05,
                2.22188918997e-05,
                7.21828018868e-06,
                6.25880011285e-06,
                3.58309898729e-06,
                3.1142340049e-06,
                3.68890596292e-06,
                3.89964952774e-06,
                1.44823593473e-06,
                9.93420030682e-07,
                9.83002450994e-07,
                8.01811609108e-07,
                6.36890490568e-07,
                6.88334239385e-07,
                6.2536923906e-07,
                5.57815155844e-07,
                6.52480309782e-07,
                8.30560509257e-07,
                9.03580826161e-07,
                7.23439029571e-07,
                5.21618460425e-07,
                5.43280748388e-07,
                4.42053448069e-07,
                2.20136058715e-07,
                1.61410582677e-07,
                2.60295337284e-07,
                2.34442921965e-07,
                1.38611582656e-07,
                9.76658824925e-08,
                7.09946093025e-08
             ],
             'max':[
                0.0475112870336,
                0.0690303221345,
                0.0259281210601,
                0.038551684469,
                0.0275548510253,
                0.0233042724431,
                0.0103824939579,
                0.00711938180029,
                0.0088077634573,
                0.00369766028598,
                0.00630062771961,
                0.00416710739955,
                0.00178902957123,
                0.000326750596287,
                0.000153029162902,
                0.000338098441716,
                0.000433359178714,
                0.0010889149271,
                0.000133697278216,
                8.931823686e-05,
                5.97970938543e-05,
                0.000101607751276,
                0.000102779922599,
                4.33854474977e-05,
                0.000106448715087,
                0.000109486369183,
                8.70572621352e-05,
                4.95615495311e-05,
                0.000143178665894,
                8.67294002092e-05,
                8.31705328892e-05,
                0.000153926463099,
                6.52171511319e-05,
                4.23371784564e-05,
                4.80573544337e-05,
                6.5673244535e-05,
                3.4235228668e-05,
                2.28322987823e-05,
                1.27902549139e-05,
                1.07691903395e-05
             ],
             'var':[
                1.6955218598e-05,
                3.54922412953e-05,
                4.84621205032e-06,
                6.29586384093e-06,
                4.16255579694e-06,
                1.67122266248e-06,
                5.86181499784e-07,
                2.51124362194e-07,
                1.7552746101e-07,
                5.79472896334e-08,
                8.5031572894e-08,
                2.09132196005e-08,
                2.730717652e-09,
                5.05658126482e-10,
                1.38541053629e-10,
                1.14987672162e-10,
                2.40534980822e-10,
                4.51966103432e-10,
                2.76249925713e-11,
                1.84969210087e-11,
                1.10940883216e-11,
                8.84241898452e-12,
                6.68899701253e-12,
                6.76353677925e-12,
                9.59350827945e-12,
                9.00797231945e-12,
                9.18322362597e-12,
                1.12003670227e-11,
                1.7176047043e-11,
                1.46558772063e-11,
                1.20022568273e-11,
                1.99215314384e-11,
                9.15082939978e-12,
                1.84780684677e-12,
                9.68140498325e-13,
                2.18293243724e-12,
                1.39230413223e-12,
                5.59351230698e-13,
                2.83650896825e-13,
                1.94888986134e-13
             ],
             'mean':[
                0.0021518394351,
                0.00368436099961,
                0.00132612208836,
                0.00155907147564,
                0.00118827645201,
                0.000565382593777,
                0.000474969914649,
                0.000244774942985,
                0.000158008595463,
                8.62608212628e-05,
                9.99855546979e-05,
                4.49473154731e-05,
                1.38626464832e-05,
                1.16330211313e-05,
                6.02314594289e-06,
                5.70184920434e-06,
                6.34836578683e-06,
                6.66369442115e-06,
                2.73947830465e-06,
                1.77635297405e-06,
                1.52758752847e-06,
                1.20802758374e-06,
                9.97662937152e-07,
                1.14644171845e-06,
                8.5787843318e-07,
                7.49789876409e-07,
                9.09917218905e-07,
                1.47154889873e-06,
                1.3535037624e-06,
                7.60884461215e-07,
                5.92455762671e-07,
                6.49110802442e-07,
                4.81578013023e-07,
                2.04892401712e-07,
                1.74970082867e-07,
                3.86064016311e-07,
                3.32383081059e-07,
                1.9672498297e-07,
                1.48688172885e-07,
                9.43669462572e-08
             ]
          },
          'spectral_complexity':{
             'dmean2':2.46444129944,
             'median':8,
             'min':0,
             'dvar2':8.90155029297,
             'dvar':3.51397848129,
             'dmean':1.48048174381,
             'max':41,
             'var':20.1335411072,
             'mean':8.33560657501
          },
          'spectral_energyband_low':{
             'dmean2':0.0116255143657,
             'median':0.00680003408343,
             'min':1.13701330614e-23,
             'dvar2':0.000676723138895,
             'dvar':0.000243530084845,
             'dmean':0.00707224663347,
             'max':0.249438390136,
             'var':0.000644486804958,
             'mean':0.0161810815334
          },
          'erbbands_crest':{
             'dmean2':4.40139627457,
             'median':12.0748519897,
             'min':1.96410489082,
             'dvar2':16.0666446686,
             'dvar':6.42769765854,
             'dmean':2.78005218506,
             'max':30.9255905151,
             'var':26.6718235016,
             'mean':12.8959417343
          },
          'silence_rate_30dB':{
             'dmean2':0.0441219173372,
             'median':1,
             'min':0,
             'dvar2':0.0505953729153,
             'dvar':0.0215720739216,
             'dmean':0.0220577567816,
             'max':1,
             'var':0.0420289486647,
             'mean':0.956035971642
          },
          'barkbands_skewness':{
             'dmean2':1.08846378326,
             'median':2.17158985138,
             'min':-2.88393449783,
             'dvar2':1.37244808674,
             'dvar':0.63165140152,
             'dmean':0.700901150703,
             'max':18.772397995,
             'var':4.26941108704,
             'mean':2.53440976143
          },
          'barkbands_flatness_db':{
             'dmean2':0.0373636446893,
             'median':0.25188177824,
             'min':0.0417264699936,
             'dvar2':0.0013774727704,
             'dvar':0.000664906052407,
             'dmean':0.0249940976501,
             'max':0.493916898966,
             'var':0.00487692654133,
             'mean':0.247923567891
          },
          'dynamic_complexity':5.36230421066,
          'mfcc':{
             'mean':[
                -769.272949219,
                174.071960449,
                24.5278339386,
                18.3546676636,
                2.7153813839,
                -11.1465549469,
                3.4257376194,
                -12.2808427811,
                -3.82589650154,
                10.8396263123,
                -13.5059509277,
                4.09915494919,
                -8.83029747009
             ],
             'icov':[
                [
                   0.000136792194098,
                   -8.43614980113e-05,
                   -2.81154789263e-05,
                   -0.000359862868208,
                   0.000170425773831,
                   -0.000186736346222,
                   -2.03711442737e-05,
                   0.000110186316306,
                   0.000183512631338,
                   -0.000137489987537,
                   0.000265641137958,
                   -0.000371046771761,
                   0.000131313965539
                ],
                [
                   -8.43614980113e-05,
                   0.000460012699477,
                   0.000123671925394,
                   -5.98719416303e-05,
                   -0.000407532555982,
                   -3.93706723116e-05,
                   5.38614440302e-05,
                   0.000160288967891,
                   -0.000176182074938,
                   -0.000304186803987,
                   0.000209134508623,
                   0.000341246428434,
                   -0.000363913859474
                ],
                [
                   -2.81154789263e-05,
                   0.000123671925394,
                   0.0023659709841,
                   -0.000972738547716,
                   -0.000427207007306,
                   -0.000387286272598,
                   8.18902117317e-05,
                   -0.000148757448187,
                   -0.00030335856718,
                   -0.00014225866471,
                   9.07063440536e-05,
                   -0.000358788878657,
                   0.000798008113634
                ],
                [
                   -0.000359862868208,
                   -5.98719416303e-05,
                   -0.000972738547716,
                   0.00611386587843,
                   -0.00256778602488,
                   -0.000126788727357,
                   -0.00104285648558,
                   -0.000827632728033,
                   -0.00087181845447,
                   0.00181972200517,
                   -0.00173351902049,
                   0.000989436171949,
                   -0.00106512603816
                ],
                [
                   0.000170425773831,
                   -0.000407532555982,
                   -0.000427207007306,
                   -0.00256778602488,
                   0.00597134232521,
                   5.03397823195e-05,
                   -0.00225527305156,
                   -0.00109422253445,
                   0.000697037321515,
                   4.05781247537e-05,
                   0.00135666213464,
                   -0.000320098013617,
                   8.51808072184e-05
                ],
                [
                   -0.000186736346222,
                   -3.93706723116e-05,
                   -0.000387286272598,
                   -0.000126788727357,
                   5.03397823195e-05,
                   0.00537943933159,
                   -0.00124698341824,
                   -0.0010569782462,
                   -0.00103715318255,
                   0.00160530698486,
                   -0.00229313620366,
                   0.00195448589511,
                   -0.00132097199094
                ],
                [
                   -2.03711442737e-05,
                   5.38614440302e-05,
                   8.18902117317e-05,
                   -0.00104285648558,
                   -0.00225527305156,
                   -0.00124698341824,
                   0.00878068804741,
                   -0.000247875228524,
                   -0.000752699153963,
                   -0.00243199244142,
                   0.00255464483052,
                   -0.000864356348757,
                   4.68621146865e-05
                ],
                [
                   0.000110186316306,
                   0.000160288967891,
                   -0.000148757448187,
                   -0.000827632728033,
                   -0.00109422253445,
                   -0.0010569782462,
                   -0.000247875228524,
                   0.00842239148915,
                   -0.00120316760149,
                   -0.00191976665519,
                   -0.00219645095058,
                   -0.000799778150395,
                   -0.00113180349581
                ],
                [
                   0.000183512631338,
                   -0.000176182074938,
                   -0.00030335856718,
                   -0.00087181845447,
                   0.000697037321515,
                   -0.00103715318255,
                   -0.000752699153963,
                   -0.00120316760149,
                   0.00913358014077,
                   -0.0011583075393,
                   -0.0034087523818,
                   -0.000684571627062,
                   -0.000547844567336
                ],
                [
                   -0.000137489987537,
                   -0.000304186803987,
                   -0.00014225866471,
                   0.00181972200517,
                   4.05781247537e-05,
                   0.00160530698486,
                   -0.00243199244142,
                   -0.00191976665519,
                   -0.0011583075393,
                   0.0099037932232,
                   -0.00213823025115,
                   -0.00240451120771,
                   0.00222806143574
                ],
                [
                   0.000265641137958,
                   0.000209134508623,
                   9.07063440536e-05,
                   -0.00173351902049,
                   0.00135666213464,
                   -0.00229313620366,
                   0.00255464483052,
                   -0.00219645095058,
                   -0.0034087523818,
                   -0.00213823025115,
                   0.012166891247,
                   -0.000136518414365,
                   -0.00396184250712
                ],
                [
                   -0.000371046771761,
                   0.000341246428434,
                   -0.000358788878657,
                   0.000989436171949,
                   -0.000320098013617,
                   0.00195448589511,
                   -0.000864356348757,
                   -0.000799778150395,
                   -0.000684571627062,
                   -0.00240451120771,
                   -0.000136518414365,
                   0.0105164479464,
                   -0.00152094080113
                ],
                [
                   0.000131313965539,
                   -0.000363913859474,
                   0.000798008113634,
                   -0.00106512603816,
                   8.51808072184e-05,
                   -0.00132097199094,
                   4.68621146865e-05,
                   -0.00113180349581,
                   -0.000547844567336,
                   0.00222806143574,
                   -0.00396184250712,
                   -0.00152094080113,
                   0.0144623927772
                ]
             ],
             'cov':[
                [
                   13856.5742188,
                   2881.48681641,
                   626.856018066,
                   985.979919434,
                   572.952941895,
                   239.108032227,
                   485.020294189,
                   -100.799766541,
                   -219.001373291,
                   156.60446167,
                   -424.508850098,
                   327.825073242,
                   -120.512145996
                ],
                [
                   2881.48681641,
                   3231.66381836,
                   134.963119507,
                   436.562103271,
                   427.08480835,
                   188.857131958,
                   212.41015625,
                   48.8245887756,
                   54.8863105774,
                   65.4324493408,
                   -54.3877830505,
                   -12.1304569244,
                   73.5505752563
                ],
                [
                   626.856018066,
                   134.963119507,
                   601.035095215,
                   273.510284424,
                   181.435073853,
                   155.473129272,
                   93.525352478,
                   110.890510559,
                   91.095451355,
                   12.8258953094,
                   64.8533477783,
                   18.9282360077,
                   27.4223709106
                ],
                [
                   985.979919434,
                   436.562103271,
                   273.510284424,
                   517.906311035,
                   277.082275391,
                   246.789306641,
                   136.501083374,
                   165.253143311,
                   136.501098633,
                   -22.0577812195,
                   126.422569275,
                   -12.4185800552,
                   100.378479004
                ],
                [
                   572.952941895,
                   427.08480835,
                   181.435073853,
                   277.082275391,
                   379.271575928,
                   107.167869568,
                   150.662460327,
                   98.3632736206,
                   44.8567466736,
                   11.3338546753,
                   7.94096279144,
                   8.45980930328,
                   33.7221221924
                ],
                [
                   239.108032227,
                   188.857131958,
                   155.473129272,
                   246.789306641,
                   107.167869568,
                   432.854797363,
                   63.7592735291,
                   167.599334717,
                   166.380645752,
                   -35.830368042,
                   191.790100098,
                   -51.9027824402,
                   122.895881653
                ],
                [
                   485.020294189,
                   212.41015625,
                   93.525352478,
                   136.501083374,
                   150.662460327,
                   63.7592735291,
                   199.520629883,
                   53.121295929,
                   32.9685554504,
                   43.6663284302,
                   -12.7093153,
                   26.8867664337,
                   8.14808177948
                ],
                [
                   -100.799766541,
                   48.8245887756,
                   110.890510559,
                   165.253143311,
                   98.3632736206,
                   167.599334717,
                   53.121295929,
                   253.994827271,
                   139.847579956,
                   39.0033912659,
                   153.251266479,
                   10.9337463379,
                   85.0499801636
                ],
                [
                   -219.001373291,
                   54.8863105774,
                   91.095451355,
                   136.501098633,
                   44.8567466736,
                   166.380645752,
                   32.9685554504,
                   139.847579956,
                   238.458724976,
                   27.9493427277,
                   166.569946289,
                   0.858494520187,
                   84.6140136719
                ],
                [
                   156.60446167,
                   65.4324493408,
                   12.8258953094,
                   -22.0577812195,
                   11.3338546753,
                   -35.830368042,
                   43.6663284302,
                   39.0033912659,
                   27.9493427277,
                   156.441299438,
                   12.6951389313,
                   54.8658180237,
                   -16.3310203552
                ],
                [
                   -424.508850098,
                   -54.3877830505,
                   64.8533477783,
                   126.422569275,
                   7.94096279144,
                   191.790100098,
                   -12.7093153,
                   153.251266479,
                   166.569946289,
                   12.6951389313,
                   260.722808838,
                   -14.3604249954,
                   111.989974976
                ],
                [
                   327.825073242,
                   -12.1304569244,
                   18.9282360077,
                   -12.4185800552,
                   8.45980930328,
                   -51.9027824402,
                   26.8867664337,
                   10.9337463379,
                   0.858494520187,
                   54.8658180237,
                   -14.3604249954,
                   133.120956421,
                   -7.61708545685
                ],
                [
                   -120.512145996,
                   73.5505752563,
                   27.4223709106,
                   100.378479004,
                   33.7221221924,
                   122.895881653,
                   8.14808177948,
                   85.0499801636,
                   84.6140136719,
                   -16.3310203552,
                   111.989974976,
                   -7.61708545685,
                   131.224090576
                ]
             ]
          },
          'spectral_decrease':{
             'dmean2':4.08620559611e-09,
             'median':-6.08908301558e-09,
             'min':-7.79806725859e-08,
             'dvar2':5.14977569259e-17,
             'dvar':1.96735594642e-17,
             'dmean':2.5918809321e-09,
             'max':3.64579226325e-17,
             'var':8.43000143983e-17,
             'mean':-8.60299742556e-09
          },
          'barkbands_crest':{
             'dmean2':4.03212404251,
             'median':12.5515785217,
             'min':2.88295245171,
             'dvar2':14.1266422272,
             'dvar':5.63002252579,
             'dmean':2.52371096611,
             'max':26.6396102905,
             'var':23.3612709045,
             'mean':13.3242540359
          }
        }
        ```
        """
        return load_extractor(self.path)["low_level"]

    @property
    def rhythm(self):
        """Rhythm: rhytm essentia extractor descriptors
        'beats_position': time positions [sec] of detected beats using beat tracking algorithm by Degara et al., 2012. Algorithms: RhythmExtractor2013, BeatTrackerDegara

        'beats_count': number of detected beats

        'bpm': BPM value according to detected beats

        'bpm_histogram_first_peak_bpm', 'bpm_histogram_first_peak_spread', 'bpm_histogram_first_peak_weight',
        'bpm_histogram_second_peak_bpm', 'bpm_histogram_second_peak_spread', 'bpm_histogram_second_peak_weight':
        descriptors characterizing highest and second highest peak of the BPM histogram. Algorithms:
        BpmHistogramDescriptors

        'beats_loudness', 'beats_loudness_band_ratio': spectral energy computed on beats segments of audio across the whole spectrum, and ratios of energy in 6 frequency bands. Algorithms: BeatsLoudness, SingleBeatLoudness

        'onset_rate': number of detected onsets per second. Algorithms: OnsetRate

        'danceability': danceability estimate. Algorithms: Danceability
        Example:
        ```JSON
        'rhythm':{
              'bpm_histogram_first_peak_bpm':{
                 'dmean2':0,
                 'median':105,
                 'min':105,
                 'dvar2':0,
                 'dvar':0,
                 'dmean':0,
                 'max':105,
                 'var':0,
                 'mean':105
              },
              'bpm':105.031990051,
              'bpm_histogram_second_peak_spread':{
                 'dmean2':0,
                 'median':0.387387394905,
                 'min':0.387387394905,
                 'dvar2':0,
                 'dvar':0,
                 'dmean':0,
                 'max':0.387387394905,
                 'var':0,
                 'mean':0.387387394905
              },
              'bpm_histogram_first_peak_weight':{
                 'dmean2':0,
                 'median':0.217948719859,
                 'min':0.217948719859,
                 'dvar2':0,
                 'dvar':0,
                 'dmean':0,
                 'max':0.217948719859,
                 'var':0,
                 'mean':0.217948719859
              },
              'onset_rate':2.74985027313,
              'beats_position':[
                 0.592108845711,
                 1.17260766029,
                 1.764716506,
                 2.34521532059,
                 2.92571425438,
                 3.48299312592,
                 4.06349182129,
                 4.63238096237,
                 5.20126962662,
                 5.7701587677,
                 6.35065746307,
                 6.91954612732,
                 7.50004529953,
                 8.08054447174,
                 8.67265319824,
                 9.26476192474,
                 9.86848068237,
                 10.4838094711,
                 11.0991382599,
                 11.7028570175,
                 12.3181858063,
                 12.933514595,
                 13.5488433838,
                 14.1757822037,
                 14.8027210236,
                 15.4412698746,
                 16.0565986633,
                 16.6719264984,
                 17.298866272,
                 17.9141941071,
                 18.5295238495,
                 19.1448516846,
                 19.7253513336,
                 20.2942390442,
                 20.8747386932,
                 21.4436283112,
                 22.0125160217,
                 22.5117454529,
                 23.010974884,
                 23.5102043152,
                 23.9978218079,
                 24.4854412079,
                 24.9730606079,
                 25.4839000702,
                 25.9831295013,
                 26.4939670563,
                 26.9931964874,
                 27.5040359497,
                 28.0032653809,
                 28.4676647186,
                 28.932062149,
                 29.3964614868,
                 29.9421310425,
                 30.476190567,
                 31.0218582153,
                 31.567527771,
                 32.1015853882,
                 32.6356468201,
                 33.1697044373,
                 33.6921539307,
                 34.2146034241,
                 34.725440979,
                 35.2478904724,
                 35.6890678406,
                 36.1302490234,
                 36.5714263916,
                 37.0126075745,
                 37.4770050049,
                 37.94140625,
                 38.4058036804,
                 38.8702049255,
                 39.334602356,
                 39.7989997864,
                 40.2634010315,
                 40.7277984619,
                 41.192199707,
                 41.7726974487,
                 42.3415870667,
                 42.9220848083,
                 43.4909744263,
                 44.0598640442,
                 44.6287536621,
                 45.1976394653,
                 45.7549209595,
                 46.3238105774,
                 46.9043083191,
                 47.4848060608,
                 48.0536956787,
                 48.6225852966,
                 49.1914749146,
                 49.7603607178,
                 50.3292503357,
                 50.8981399536,
                 51.4786376953,
                 52.0475273132,
                 52.6164169312,
                 53.1969146729,
                 53.7658042908,
                 54.3346939087,
                 54.9151916504,
                 55.4956893921,
                 56.0761909485,
                 56.6334686279,
                 57.1907463074,
                 57.7480278015,
                 58.3285255432,
                 58.8974151611,
                 59.4663009644,
                 60.0468025208,
                 60.6156921387,
                 61.1845779419,
                 61.7534675598,
                 62.3223571777,
                 62.8912467957,
                 63.4601364136,
                 64.0406341553,
                 64.621131897,
                 65.1900177002,
                 65.7589111328,
                 66.3394088745,
                 66.9082946777,
                 67.4771881104,
                 68.0576858521,
                 68.6265716553,
                 69.1954650879,
                 69.7759628296,
                 70.3448486328,
                 70.9137420654,
                 71.4942398071,
                 72.0515213013,
                 72.6204071045,
                 73.1892929077,
                 73.7581863403,
                 74.3270721436,
                 74.8959655762,
                 75.4764633179,
                 76.0453491211,
                 76.6142425537,
                 77.1947402954,
                 77.7636260986,
                 78.3209075928,
                 78.9014053345,
                 79.4819030762,
                 80.0624008179,
                 80.619682312,
                 81.1769561768,
                 81.7574615479,
                 82.3263473511,
                 82.8952331543,
                 83.4525146484,
                 84.0330123901,
                 84.6019058228,
                 85.170791626,
                 85.7512893677,
                 86.3317871094,
                 86.9122848511,
                 87.4927902222,
                 88.0616760254,
                 88.6305618286,
                 89.1994552612,
                 89.7799530029,
                 90.3604507446,
                 90.9293441772,
                 91.4982299805,
                 92.0671157837,
                 92.6360092163,
                 93.2048950195,
                 93.7737884521,
                 94.3426742554,
                 94.8999557495,
                 95.4688415527,
                 96.0493392944,
                 96.6066207886,
                 97.1755065918,
                 97.7560043335,
                 98.3365097046,
                 98.9053955078,
                 99.474281311,
                 100.043174744,
                 100.600448608,
                 101.169342041,
                 101.726615906,
                 102.295509338,
                 102.725074768,
                 103.166259766,
                 103.595825195,
                 104.025398254,
                 104.466575623,
                 104.907752991,
                 105.33732605,
                 105.766891479,
                 106.208068848,
                 106.649246216,
                 107.067207336,
                 107.485168457,
                 107.914733887,
                 108.344306946,
                 108.762268066,
                 109.191833496,
                 109.621406555,
                 110.062583923,
                 110.480545044,
                 110.898498535,
                 111.328071594,
                 111.757637024,
                 112.175598145,
                 112.605171204,
                 113.034736633,
                 113.452697754,
                 113.882263184,
                 114.311836243,
                 114.753013611,
                 115.194190979,
                 115.623764038,
                 116.041717529,
                 116.471290588,
                 116.912467957,
                 117.330429077,
                 117.748390198,
                 118.177955627,
                 118.607528687,
                 119.048706055,
                 119.478271484,
                 119.907844543,
                 120.325805664,
                 120.755371094,
                 121.184944153,
                 121.614509583,
                 122.044082642,
                 122.473648071,
                 122.90322113,
                 123.33278656,
                 123.762359619,
                 124.18031311,
                 124.609886169,
                 125.039451599,
                 125.480628967,
                 125.910202026,
                 126.339767456,
                 126.780952454,
                 127.198905945,
                 127.616867065,
                 128.046432495,
                 128.476013184,
                 128.905578613,
                 129.335144043,
                 129.764709473,
                 130.194290161,
                 130.623855591,
                 131.053421021,
                 131.48298645,
                 131.912567139,
                 132.342132568,
                 132.771697998,
                 133.212875366,
                 133.642440796,
                 134.072021484,
                 134.501586914,
                 134.931152344,
                 135.349105835,
                 135.767074585,
                 136.196640015,
                 136.626205444,
                 137.044174194,
                 137.473739624,
                 137.903305054,
                 138.344482422,
                 138.77406311,
                 139.20362854,
                 139.621582031,
                 140.039550781,
                 140.469116211,
                 140.898681641,
                 141.32824707,
                 141.757827759,
                 142.187393188,
                 142.616958618,
                 143.058135986,
                 143.487701416,
                 143.917282104,
                 144.335235596,
                 144.753189087,
                 145.182769775,
                 145.600723267,
                 146.065124512,
                 146.529525757,
                 147.005523682,
                 147.446716309,
                 147.876281738,
                 148.305847168,
                 148.758636475,
                 149.211425781,
                 149.664215088,
                 150.128616333,
                 150.593017578,
                 151.057418823,
                 151.52180481,
                 152.009429932,
                 152.485443115,
                 152.973052979,
                 153.414230347,
                 153.843811035,
                 154.273376465,
                 154.761001587,
                 155.24861145,
                 155.736236572,
                 156.235458374,
                 156.734695435,
                 157.245529175,
                 157.756362915,
                 158.267211914,
                 158.778045654,
                 159.288879395
              ],
              'beats_loudness':{
                 'dmean2':0.0512414947152,
                 'median':0.030088307336,
                 'min':0,
                 'dvar2':0.00452459603548,
                 'dvar':0.0014941196423,
                 'dmean':0.0283653773367,
                 'max':0.2241974473,
                 'var':0.00184174487367,
                 'mean':0.0413691326976
              },
              'danceability':0.971863210201,
              'bpm_histogram_second_peak_weight':{
                 'dmean2':0,
                 'median':0.217948719859,
                 'min':0.217948719859,
                 'dvar2':0,
                 'dvar':0,
                 'dmean':0,
                 'max':0.217948719859,
                 'var':0,
                 'mean':0.217948719859
              },
              'beats_count':313,
              'bpm_histogram_second_peak_bpm':{
                 'dmean2':0,
                 'median':140,
                 'min':140,
                 'dvar2':0,
                 'dvar':0,
                 'dmean':0,
                 'max':140,
                 'var':0,
                 'mean':140
              },
              'bpm_histogram_first_peak_spread':{
                 'dmean2':0,
                 'median':0.464566975832,
                 'min':0.464566975832,
                 'dvar2':0,
                 'dvar':0,
                 'dmean':0,
                 'max':0.464566975832,
                 'var':0,
                 'mean':0.464566975832
              },
              'beats_loudness_band_ratio':{
                 'dmean2':[
                    0.293352067471,
                    0.333939641714,
                    0.216318577528,
                    0.0560376681387,
                    0.00940202735364,
                    0.0521685443819
                 ],
                 'median':[
                    0.462669193745,
                    0.257794469595,
                    0.105961687863,
                    0.0136956917122,
                    0.00351257342845,
                    0.00198904867284
                 ],
                 'min':[
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                 ],
                 'dvar2':[
                    0.0780283510685,
                    0.0940517783165,
                    0.0547315329313,
                    0.00607642298564,
                    0.000168528946233,
                    0.0274331402034
                 ],
                 'dvar':[
                    0.0290206372738,
                    0.0327779129148,
                    0.0200501643121,
                    0.00221319519915,
                    6.26860273769e-05,
                    0.00980541761965
                 ],
                 'dmean':[
                    0.170528307557,
                    0.192360565066,
                    0.126187071204,
                    0.031432043761,
                    0.00537649122998,
                    0.0285477209836
                 ],
                 'max':[
                    0.944636106491,
                    0.987443983555,
                    0.971004068851,
                    0.404173195362,
                    0.0544534698129,
                    0.777507662773
                 ],
                 'var':[
                    0.0980709269643,
                    0.072200588882,
                    0.0500059388578,
                    0.00271857017651,
                    6.26871405984e-05,
                    0.00670601474121
                 ],
                 'mean':[
                    0.450265616179,
                    0.34460362792,
                    0.196296930313,
                    0.0317945256829,
                    0.00625067576766,
                    0.0202882513404
                 ]
              }
           },
           ```
        """
        return load_extractor(self.path)["metadata"]["rhythm"]

    def to_jams(self):
        """Jams: the track's data in jams format"""
        return jams_utils.jams_converter(
            metadata={
                'features': load_extractor(self.path),
                'duration': load_extractor(self.path)["metadata"]["audio_properties"]["length"]
            }
        )


def load_extractor(path):
    """Load a AcousticBrainz Dataset json file with all the features and metadata.

    Args:
        path (str): path to features and metadata path

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    if not os.path.exists(path):
        raise IOError("path {} does not exist".format(path))

    with open(path) as json_file:
        meta = json.load(json_file)
    return meta


def filter_index(search_key, index=None):
    """Load from AcousticBrainz genre dataset the indexes that match with search_key.

    Args:
        search_key (str): regex to match with folds, mbid or genres
        index (dict): mirdata index to filter.

    Returns:
        (dict): {`track_id`: track data}
    """
    if index is None:
        index = DATA.index["tracks"].items()

    acousticbrainz_genre_data = {k: v for k, v in index.items() if search_key in k}
    return acousticbrainz_genre_data


def load_all_train(index=None):
    """Load from AcousticBrainz genre dataset the tracks that are used for training across the four different datasets.

        Args:
            index (dict): mirdata index to filter.

        Returns:
            (dict): {`track_id`: track data}

    """
    return filter_index("#train#", index=index)


def load_all_validation(index=None):
    """Load from AcousticBrainz genre dataset the tracks that are used for validating across the four different datasets.

            Args:
                index (dict): mirdata index to filter.

            Returns:
                (dict): {`track_id`: track data}

    """
    return filter_index("#validation#", index=index)


def load_tagtraum_validation(index=None):
    """Load from AcousticBrainz genre dataset the tracks that are used for validating in tagtraum dataset.

                Args:
                    data_home (str): Local path where the dataset is stored.
                        If `None`, looks for the data in the default directory, `~/mir_datasets`
                    index (dict): mirdata index to filter.

                Returns:
                    (dict): {`track_id`: track data}

    """
    return filter_index("tagtraum#validation#", index=index)


def load_tagtraum_train(index=None):
    """Load from AcousticBrainz genre dataset the tracks that are used for training in tagtraum dataset.

                    Args:
                        index (dict): mirdata index to filter.

                    Returns:
                        (dict): {`track_id`: track data}

    """
    return filter_index("tagtraum#train#", index=index)


def load_allmusic_train(index=None):
    """Load from AcousticBrainz genre dataset the tracks that are used for validation in allmusic dataset.

                    Args:
                        index (dict): mirdata index to filter.

                    Returns:
                        (dict): {`track_id`: track data}

    """
    return filter_index("allmusic#train#", index=index)


def load_allmusic_validation(index=None):
    """Load from AcousticBrainz genre dataset the tracks that are used for validation in allmusic dataset.

                    Args:
                        index (dict): mirdata index to filter.

                    Returns:
                        (dict): {`track_id`: track data}

    """
    return filter_index("allmusic#validation#", index=index)


def load_lastfm_train(index=None):
    """Load from AcousticBrainz genre dataset the tracks that are used for training in lastfm dataset.

                    Args:
                        index (dict): mirdata index to filter.

                    Returns:
                        (dict): {`track_id`: track data}

    """
    return filter_index("lastfm#train#", index=index)


def load_lastfm_validation(index=None):
    """Load from AcousticBrainz genre dataset the tracks that are used for validation in lastfm dataset.

                    Args:
                        index (dict): mirdata index to filter.

                    Returns:
                        (dict): {`track_id`: track data}

    """
    return filter_index("lastfm#validation#", index=index)


def load_discogs_train(index=None):
    """Load from AcousticBrainz genre dataset the tracks that are used for training in discogs dataset.

                    Args:
                        index (dict): mirdata index to filter.

                    Returns:
                        (dict): {`track_id`: track data}

    """
    return filter_index("allmusic#train#", index=index)


def load_discogs_validation(index=None):
    """Load from AcousticBrainz genre dataset the tracks that are used for validation in tagtraum dataset.

                    Args:
                        index (dict): mirdata index to filter.

                    Returns:
                        (dict): {`track_id`: track data}

    """
    return filter_index("allmusic#validation#", index=index)


def _download(save_dir, remotes, partial_download, info_message, force_overwrite=False, cleanup=True):
    """Download data to `save_dir` and optionally print a message.

    Args:
        save_dir (str):
            Dataset files path
            remotes (dict or None):
        remotes (dict) :
            A dictionary of RemoteFileMetadata tuples of data in zip format.
            If None, there is no data to download
        force_overwrite (bool):
                If True, existing files are overwritten by the downloaded files. By default False.
        cleanup (bool):
            Whether to delete any zip/tar files after extracting.

    Raises:
        ValueError: if invalid keys are passed to partial_download
        IOError: if a downloaded file's checksum is different from expected

    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Create these directories if doesn't exist
    train = "acousticbrainz-mediaeval-train"
    train_dir = os.path.join(save_dir, train)
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    validate = "acousticbrainz-mediaeval-validation"
    validate_dir = os.path.join(save_dir, validate)
    if not os.path.isdir(validate_dir):
        os.mkdir(validate_dir)

    # start to download
    for key, remote in remotes.items():
        # check overwrite
        file_downloaded = False
        if not force_overwrite:
            fold, first_dir = key.split('-')
            first_dir_path = os.path.join(train_dir if fold == 'train' else validate_dir, first_dir)
            if os.path.isdir(first_dir_path):
                file_downloaded = True
                print("File " + remote.filename + " downloaded. Skip download (force_overwrite=False).")
        if not file_downloaded:
                #  if this typical error happend it repeat download
                download_utils.downloader(
                    save_dir,
                    remotes={key: remote},
                    partial_download=None,
                    info_message=None,
                    force_overwrite=True,
                    cleanup=cleanup,
                )
        # move from a temporal directory to final one
        source_dir = os.path.join(save_dir, "temp", train if "train" in key else validate)
        target_dir = train_dir if "train" in key else validate_dir
        dir_names = os.listdir(source_dir)
        for dir_name in dir_names:
            shutil.move(os.path.join(source_dir, dir_name), os.path.join(target_dir, dir_name))
