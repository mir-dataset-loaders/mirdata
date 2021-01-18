# -*- coding: utf-8 -*-
"""Acoustic Brainz Genre dataset

.. admonition:: Dataset Info
    :class: dropdown

    The AcousticBrainz Genre Dataset consists of four datasets of genre annotations and music features extracted from audio
    suited for evaluation of hierarchical multi-label genre classification systems.

    Description about the music features can be found here: https://essentia.upf.edu/streaming_extractor_music.html

    The datasets are used within the MediaEval AcousticBrainz Genre Task. The task is focused on content-based music
    genre recognition using genre annotations from multiple sources and large-scale music features data available in the
    AcousticBrainz database. The goal of our task is to explore how the same music pieces can be annotated differently by
    different communities following different genre taxonomies, and how this should be addressed by content-based genre r
    ecognition systems.

    We provide four datasets containing genre and subgenre annotations extracted from four different online metadata sources:

    - AllMusic and Discogs are based on editorial metadata databases maintained by music experts and enthusiasts. These sources 
      contain explicit genre/subgenre annotations of music releases (albums) following a predefined genre namespace and taxonomy. 
      We propagated release-level annotations to recordings (tracks) in AcousticBrainz to build the datasets.
    - Lastfm and Tagtraum are based on collaborative music tagging platforms with large amounts of genre labels provided by their 
      users for music recordings (tracks). We have automatically inferred a genre/subgenre taxonomy and annotations from these labels.

    For details on format and contents, please refer to the data webpage.

    Note, that the AllMusic ground-truth annotations are distributed separately at https://zenodo.org/record/2554044.

    If you use the MediaEval AcousticBrainz Genre dataset or part of it, please cite our ISMIR 2019 overview paper:

    .. code-block:: latex

        Bogdanov, D., Porter A., Schreiber H., Urbano J., & Oramas S. (2019).
        The AcousticBrainz Genre Dataset: Multi-Source, Multi-Level, Multi-Label, and Large-Scale.
        20th International Society for Music Information Retrieval Conference (ISMIR 2019).

    This work is partially supported by the European Union’s Horizon 2020 research and innovation programme under 
    grant agreement No 688382 AudioCommons.

"""

import json
import os
import shutil

from mirdata import download_utils, core
from mirdata import jams_utils


BIBTEX = """
@inproceedings{bogdanov2019acousticbrainz,
  title={The AcousticBrainz genre dataset: Multi-source, multi-level, multi-label, and large-scale},
  author={Bogdanov, Dmitry and Porter, Alastair and Schreiber, Hendrik and Urbano, Juli{\'a}n and Oramas, Sergio},
  booktitle={Proceedings of the 20th Conference of the International Society for Music Information Retrieval (ISMIR 2019): 2019 Nov 4-8; Delft, The Netherlands.[Canada]: ISMIR; 2019.},
  year={2019},
  organization={International Society for Music Information Retrieval (ISMIR)}
}
"""
REMOTES = {
    "validation-01": download_utils.RemoteFileMetadata(
        filename="acousticbrainz-mediaeval-features-validation-01234567.tar.bz2",
        url="https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-validation-01234567.tar.bz2?download=1",
        checksum="f21f9c5e398713139cca9790b656faf9",
        destination_dir="temp",
    ),
    "validation-89": download_utils.RemoteFileMetadata(
        filename="acousticbrainz-mediaeval-features-validation-89abcdef.tar.bz2",
        url="https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-validation-89abcdef.tar.bz2?download=1",
        checksum="34f47394ac6d8face4399f48e2b98ebe",
        destination_dir="temp",
    ),
    "train-01": download_utils.RemoteFileMetadata(
        filename="acousticbrainz-mediaeval-features--train-01.tar.bz2",
        url="https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features--train-01.tar.bz2?download=1",
        checksum="db7157b5112022d609652dd21c632090",
        destination_dir="temp",
    ),
    "train-23": download_utils.RemoteFileMetadata(
        filename="acousticbrainz-mediaeval-features-train-23.tar.bz2",
        url="https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-train-23.tar.bz2?download=1",
        checksum="79581967a1be5c52e83be21261d1ef6c",
        destination_dir="temp",
    ),
    "train-45": download_utils.RemoteFileMetadata(
        filename="acousticbrainz-mediaeval-features-train-45.tar.bz2",
        url="https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-train-45.tar.bz2?download=1",
        checksum="0e48fa319fa48e5cf95eea8118d2e882",
        destination_dir="temp",
    ),
    "train-67": download_utils.RemoteFileMetadata(
        filename="acousticbrainz-mediaeval-features-train-67.tar.bz2",
        url="https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-train-67.tar.bz2?download=1",
        checksum="22ca7f1fea8a86459b7fda4530f00070",
        destination_dir="temp",
    ),
    "train-89": download_utils.RemoteFileMetadata(
        filename="acousticbrainz-mediaeval-features-train-89.tar.bz2",
        url="https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-train-89.tar.bz2?download=1",
        checksum="c6e4a2ef1b0e8ed535197b868f8c7302",
        destination_dir="temp",
    ),
    "train-ab": download_utils.RemoteFileMetadata(
        filename="acousticbrainz-mediaeval-features-train-ab.tar.bz2",
        url="https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-train-ab.tar.bz2?download=1",
        checksum="513d5f306dd4f3799c137423ee444051",
        destination_dir="temp",
    ),
    "train-cd": download_utils.RemoteFileMetadata(
        filename="acousticbrainz-mediaeval-features-train-cd.tar.bz2",
        url="https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-train-cd.tar.bz2?download=1",
        checksum="422d75d70d583decec0b2761865092a7",
        destination_dir="temp",
    ),
    "train-ef": download_utils.RemoteFileMetadata(
        filename="acousticbrainz-mediaeval-features-train-ef.tar.bz2",
        url="https://zenodo.org/record/2553414/files/acousticbrainz-mediaeval-features-train-ef.tar.bz2?download=1",
        checksum="021ab25a5fd1b020521824e7fce9c775",
        destination_dir="temp",
    ),
}
REMOTE_INDEX = {
    "REMOTE_INDEX": download_utils.RemoteFileMetadata(
        filename="acousticbrainz_genre_index.json.zip",
        url="https://zenodo.org/record/4298580/files/acousticbrainz_genre_index.json.zip?download=1",
        checksum="810f1c003f53cbe58002ba96e6d4d138",
        destination_dir="",
    )
}

DATA = core.LargeData("acousticbrainz_genre_index.json", remote_index=REMOTE_INDEX)

LICENSE_INFO = """
This dataset is composed of 4 subdatasets. Three of them are Creative Commons Attribution 
Non Commercial Share Alike 4.0 International and the other one is non-comercial. Details 
about which license correspond to each subdataset can be found in the following websites:

* https://zenodo.org/record/2553414#.X_nxnOn7RUI 
* https://zenodo.org/record/2554044#.X_nw2en7RUI
 
 """


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
            data = core.LargeData(remote_index_name, remote_index=remote_index)
        else:
            data = DATA

        if track_id not in data.index["tracks"]:
            raise ValueError(
                "{} is not a valid track ID in AcousticBrainz genre Dataset".format(
                    track_id
                )
            )

        self.track_id = track_id
        self._data_home = data_home
        self._track_paths = data.index["tracks"][track_id]
        self.path = core.none_path_join([self._data_home, self._track_paths["data"][0]])

    # Genre
    @property
    def genre(self):
        """human-labeled genre and subgenres list

        Returns:
            list: human-labeled genre and subgenres list

        """
        return [genre for genre in self.track_id.split("#")[2:]]

    # Music Brainz
    @property
    def mbid(self):
        """musicbrainz id

        Returns:
            str: mbid

        """
        return self.track_id.split("#")[0]

    @property
    def mbid_group(self):
        """musicbrainz id group

        Returns:
            str: mbid group

        """
        return self.track_id.split("#")[1]

    # Metadata
    @property
    def artist(self):
        """metadata artist annotation

        Returns:
            str: artist

        """
        return load_extractor(self.path)["metadata"]["artist"]

    @property
    def title(self):
        """metadata title annotation

        Returns:
            str: title

        """
        return load_extractor(self.path)["metadata"]["title"]

    @property
    def date(self):
        """metadata date annotation

        Returns:
            str: date

        """
        return load_extractor(self.path)["metadata"]["date"]

    @property
    def file_name(self):
        """metadata file_name annotation

        Returns:
            str: file name
        """
        return load_extractor(self.path)["metadata"]["file_name"]

    @property
    def album(self):
        """metadata album annotation

        Returns:
            str: album
        """
        return load_extractor(self.path)["metadata"]["album"]

    @property
    def tracknumber(self):
        """metadata tracknumber annotation

        Returns:
            str: tracknumber
        """
        return load_extractor(self.path)["metadata"]["tracknumber"]

    @property
    def tonal(self):
        """tonal features

        Returns:
            dict:
            .. toggle::

                - 'tuning_frequency': estimated tuning frequency [Hz]. Algorithms: TuningFrequency
                - 'tuning_nontempered_energy_ratio' and 'tuning_equal_tempered_deviation'
                - 'hpcp', 'thpcp': 32-dimensional harmonic pitch class profile (HPCP) and its transposed version. Algorithms: HPCP
                - 'hpcp_entropy': Shannon entropy of a HPCP vector. Algorithms: Entropy
                - 'key_key', 'key_scale': Global key feature. Algorithms: Key
                - 'chords_key', 'chords_scale': Global key extracted from chords detection.
                - 'chords_strength', 'chords_histogram': : strength of estimated chords and normalized histogram of their
                  progression; Algorithms: ChordsDetection, ChordsDescriptors
                - 'chords_changes_rate', 'chords_number_rate':  chords change rate in the progression; ratio
                  of different chords from the total number of chords in the progression; Algorithms: ChordsDetection,
                  ChordsDescriptors

        """
        return load_extractor(self.path)["tonal"]

    @property
    def low_level(self):
        """low_level track descritors.

        Returns:
           dict:
           .. toggle::

                - 'average_loudness': dynamic range descriptor. It rescales average loudness,
                  computed on 2sec windows with 1 sec overlap, into the [0,1] interval. The value of 0 corresponds to signals
                  with large dynamic range, 1 corresponds to signal with little dynamic range. Algorithms: Loudness
                - 'dynamic_complexity': dynamic complexity computed on 2sec windows with 1sec overlap. Algorithms: DynamicComplexity
                - 'silence_rate_20dB', 'silence_rate_30dB', 'silence_rate_60dB': rate of silent frames in a signal for
                  thresholds of 20, 30, and 60 dBs. Algorithms: SilenceRate
                - 'spectral_rms': spectral RMS. Algorithms: RMS
                - 'spectral_flux': spectral flux of a signal computed using L2-norm. Algorithms: Flux
                - 'spectral_centroid', 'spectral_kurtosis', 'spectral_spread', 'spectral_skewness': centroid and central
                  moments statistics describing the spectral shape. Algorithms: Centroid, CentralMoments
                - 'spectral_rolloff': the roll-off frequency of a spectrum. Algorithms: RollOff
                - 'spectral_decrease': spectral decrease. Algorithms: Decrease
                - 'hfc': high frequency content descriptor as proposed by Masri. Algorithms: HFC
                - 'zerocrossingrate' zero-crossing rate. Algorithms: ZeroCrossingRate
                - 'spectral_energy': spectral energy. Algorithms: Energy
                - 'spectral_energyband_low', 'spectral_energyband_middle_low', 'spectral_energyband_middle_high',
                - 'spectral_energyband_high': spectral energy in frequency bands [20Hz, 150Hz], [150Hz, 800Hz], [800Hz, 4kHz],
                  and [4kHz, 20kHz]. Algorithms EnergyBand
                - 'barkbands': spectral energy in 27 Bark bands. Algorithms: BarkBands
                - 'melbands': spectral energy in 40 mel bands. Algorithms: MFCC
                - 'erbbands': spectral energy in 40 ERB bands. Algorithms: ERBBands
                - 'mfcc': the first 13 mel frequency cepstrum coefficients. See algorithm: MFCC
                - 'gfcc': the first 13 gammatone feature cepstrum coefficients. Algorithms: GFCC
                - 'barkbands_crest', 'barkbands_flatness_db': crest and flatness computed over energies in Bark bands. Algorithms: Crest, FlatnessDB
                - 'barkbands_kurtosis', 'barkbands_skewness', 'barkbands_spread': central moments statistics over energies in Bark bands. Algorithms: CentralMoments
                - 'melbands_crest', 'melbands_flatness_db': crest and flatness computed over energies in mel bands. Algorithms: Crest, FlatnessDB
                - 'melbands_kurtosis', 'melbands_skewness', 'melbands_spread': central moments statistics over energies in mel bands. Algorithms: CentralMoments
                - 'erbbands_crest', 'erbbands_flatness_db': crest and flatness computed over energies in ERB bands. Algorithms: Crest, FlatnessDB
                - 'erbbands_kurtosis', 'erbbands_skewness', 'erbbands_spread': central moments statistics over energies in ERB bands. Algorithms: CentralMoments
                - 'dissonance': sensory dissonance of a spectrum. Algorithms: Dissonance
                - 'spectral_entropy': Shannon entropy of a spectrum. Algorithms: Entropy
                - 'pitch_salience': pitch salience of a spectrum. Algorithms: PitchSalience
                - 'spectral_complexity': spectral complexity. Algorithms: SpectralComplexity
                - 'spectral_contrast_coeffs', 'spectral_contrast_valleys': spectral contrast features. Algorithms:
                  SpectralContrast

        """
        return load_extractor(self.path)["low_level"]

    @property
    def rhythm(self):
        """rhythm essentia extractor descriptors

        Returns:
             dict:
             .. toggle::

                - 'beats_position': time positions [sec] of detected beats using beat tracking algorithm by Degara et al., 2012. Algorithms: RhythmExtractor2013, BeatTrackerDegara
                - 'beats_count': number of detected beats
                - 'bpm': BPM value according to detected beats
                - 'bpm_histogram_first_peak_bpm', 'bpm_histogram_first_peak_spread', 'bpm_histogram_first_peak_weight',
                - 'bpm_histogram_second_peak_bpm', 'bpm_histogram_second_peak_spread', 'bpm_histogram_second_peak_weight':
                  descriptors characterizing highest and second highest peak of the BPM histogram. Algorithms:
                  BpmHistogramDescriptors
                - 'beats_loudness', 'beats_loudness_band_ratio': spectral energy computed on beats segments of audio
                  across the whole spectrum, and ratios of energy in 6 frequency bands.
                  Algorithms: BeatsLoudness, SingleBeatLoudness
                - 'onset_rate': number of detected onsets per second. Algorithms: OnsetRate
                - 'danceability': danceability estimate. Algorithms: Danceability
        """
        return load_extractor(self.path)["metadata"]["rhythm"]

    def to_jams(self):
        """the track's data in jams format

        Returns:
             jams.JAMS: return track data in jam format

        """
        return jams_utils.jams_converter(
            metadata={
                "features": load_extractor(self.path),
                "duration": load_extractor(self.path)["metadata"]["audio_properties"][
                    "length"
                ],
            }
        )


def load_extractor(path):
    """Load a AcousticBrainz Dataset json file with all the features and metadata.

    Args:
        path (str): path to features and metadata path

    Returns:
        * np.ndarray - the mono audio signal
        * float - The sample rate of the audio file

    """
    if not os.path.exists(path):
        raise IOError("path {} does not exist".format(path))

    with open(path) as json_file:
        meta = json.load(json_file)
    return meta


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The acousticbrainz genre dataset
    """

    def __init__(self, data_home=None, index=None):
        super().__init__(
            data_home,
            index=DATA.index if index is None else index,
            name="acousticbrainz_genre",
            track_object=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            license_info=LICENSE_INFO,
        )

    @core.copy_docs(load_extractor)
    def load_extractor(self, *args, **kwargs):
        return load_extractor(*args, **kwargs)

    def download(self, partial_download=None, force_overwrite=False, cleanup=False):
        """Download the dataset

        Args:
            partial_download (list or None):
                A list of keys of remotes to partially download.
                If None, all data is downloaded
            force_overwrite (bool):
                If True, existing files are overwritten by the downloaded files.
                By default False.
            cleanup (bool):
                Whether to delete any zip/tar files after extracting.

        Raises:
            ValueError: if invalid keys are passed to partial_download
            IOError: if a downloaded file's checksum is different from expected

        """
        if not os.path.exists(self.data_home):
            os.makedirs(self.data_home)
        # Create these directories if doesn't exist
        train = "acousticbrainz-mediaeval-train"
        train_dir = os.path.join(self.data_home, train)
        if not os.path.isdir(train_dir):
            os.mkdir(train_dir)
        validate = "acousticbrainz-mediaeval-validation"
        validate_dir = os.path.join(self.data_home, validate)
        if not os.path.isdir(validate_dir):
            os.mkdir(validate_dir)

        # start to download
        for key, remote in self.remotes.items():
            # check overwrite
            file_downloaded = False
            if not force_overwrite:
                fold, first_dir = key.split("-")
                first_dir_path = os.path.join(
                    train_dir if fold == "train" else validate_dir, first_dir
                )
                if os.path.isdir(first_dir_path):
                    file_downloaded = True
                    print(
                        "File "
                        + remote.filename
                        + " downloaded. Skip download (force_overwrite=False)."
                    )
            if not file_downloaded:
                #  if this typical error happend it repeat download
                download_utils.downloader(
                    self.data_home,
                    remotes={key: remote},
                    partial_download=None,
                    info_message=None,
                    force_overwrite=True,
                    cleanup=cleanup,
                )
            # move from a temporary directory to final one
            source_dir = os.path.join(
                self.data_home, "temp", train if "train" in key else validate
            )
            target_dir = train_dir if "train" in key else validate_dir
            dir_names = os.listdir(source_dir)
            for dir_name in dir_names:
                shutil.move(
                    os.path.join(source_dir, dir_name),
                    os.path.join(target_dir, dir_name),
                )

    def filter_index(self, search_key):
        """Load from AcousticBrainz genre dataset the indexes that match with search_key.

        Args:
            search_key (str): regex to match with folds, mbid or genres

        Returns:
             dict: {`track_id`: track data}

        """

        acousticbrainz_genre_data = {
            k: v for k, v in self._index["tracks"].items() if search_key in k
        }
        return acousticbrainz_genre_data

    def load_all_train(self):
        """Load from AcousticBrainz genre dataset the tracks that are used for training across the four different datasets.

        Returns:
            dict: {`track_id`: track data}

        """
        return self.filter_index("#train#")

    def load_all_validation(self):
        """Load from AcousticBrainz genre dataset the tracks that are used for validating across the four different datasets.

        Returns:
            dict: {`track_id`: track data}

        """
        return self.filter_index("#validation#")

    def load_tagtraum_validation(self):
        """Load from AcousticBrainz genre dataset the tracks that are used for validating in tagtraum dataset.

        Returns:
            dict: {`track_id`: track data}

        """
        return self.filter_index("tagtraum#validation#")

    def load_tagtraum_train(self):
        """Load from AcousticBrainz genre dataset the tracks that are used for training in tagtraum dataset.

        Returns:
            dict: {`track_id`: track data}

        """
        return self.filter_index("tagtraum#train#")

    def load_allmusic_train(self):
        """Load from AcousticBrainz genre dataset the tracks that are used for validation in allmusic dataset.

        Returns:
            dict: {`track_id`: track data}

        """
        return self.filter_index("allmusic#train#")

    def load_allmusic_validation(self):
        """Load from AcousticBrainz genre dataset the tracks that are used for validation in allmusic dataset.

        Returns:
            dict: {`track_id`: track data}

        """
        return self.filter_index("allmusic#validation#")

    def load_lastfm_train(self):
        """Load from AcousticBrainz genre dataset the tracks that are used for training in lastfm dataset.

        Returns:
            dict: {`track_id`: track data}

        """
        return self.filter_index("lastfm#train#")

    def load_lastfm_validation(self):
        """Load from AcousticBrainz genre dataset the tracks that are used for validation in lastfm dataset.

        Returns:
            dict: {`track_id`: track data}

        """
        return self.filter_index("lastfm#validation#")

    def load_discogs_train(self):
        """Load from AcousticBrainz genre dataset the tracks that are used for training in discogs dataset.

        Returns:
            dict: {`track_id`: track data}

        """
        return self.filter_index("allmusic#train#")

    def load_discogs_validation(self):
        """Load from AcousticBrainz genre dataset the tracks that are used for validation in tagtraum dataset.

        Returns:
            dict: {`track_id`: track data}

        """
        return self.filter_index("allmusic#validation#")
