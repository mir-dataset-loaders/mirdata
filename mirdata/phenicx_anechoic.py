# -*- coding: utf-8 -*-
"""PHENICX-Anechoic Dataset Loader

This dataset includes audio and annotations useful for tasks as score-informed source separation, score following, multi-pitch estimation, transcription or instrument detection, in the context of symphonic music:
M. Miron, J. Carabias-Orti, J. J. Bosch, E. Gómez and J. Janer, "Score-informed source separation for multi-channel orchestral recordings", Journal of Electrical and Computer Engineering (2016))"

We do not provide the original audio files, which can be found at the web page hosted by Aalto University. However, with their permission we distribute the denoised versions for some of the anechoic orchestral recordings. The original dataset was introduced in:
Pätynen, J., Pulkki, V., and Lokki, T., "Anechoic recording system for symphony orchestra," Acta Acustica united with Acustica, vol. 94, nr. 6, pp. 856-865, November/December 2008.

Additionally, we provide the associated musical note onset and offset annotations, and the Roomsim configuration files used to generate the multi-microphone recordings.

The original anechoic dataset in Pätynen et al. consists of four passages of symphonic music from the Classical and Romantic periods. This work presented a set of anechoic recordings for each of the instruments, which were then synchronized between them so that they could later be combined to a mix of the orchestra. In order to keep the evaluation setup consistent between the four pieces, we selected the following instruments: violin, viola, cello, double bass, oboe, flute, clarinet, horn, trumpet and bassoon. A list of the characteristics of the four pieces can be found below:

Mozart
- duration: 3min 47s
- period: classical
- no. sources: 8
- total no. instruments: 10
- max. instruments/source: 2

Beethoven
- duration: 3min 11s
- period: classical
- no. sources: 10
- total no. instruments: 20
- max. instruments/source: 4

Beethoven
- duration: 2min 12s
- period: romantic
- no. sources: 10
- total no. instruments: 30
- max. instruments/source: 4

Bruckner
- duration: 1min 27s
- period: romantic
- no. sources: 10
- total no. instruments: 39
- max. instruments/source: 12

For more details, please visit: https://www.upf.edu/web/mtg/phenicx-anechoic

"""


import csv
import glob
import librosa
import logging
import numpy as np
import os
import shutil
import collections
import string

from mirdata import download_utils
from mirdata import jams_utils
from mirdata import track
from mirdata import utils

# -- REMOTES is a dictionary containing all files that need to be downloaded.
# -- The keys should be descriptive (e.g. 'annotations', 'audio')
REMOTES = {
    'all': download_utils.RemoteFileMetadata(
        filename='PHENICX-Anechoic.zip',
        url='https://zenodo.org/record/840025/files/PHENICX-Anechoic.zip?download=1',
        checksum='7fec47568263476ecac0103aef608629',
        destination_dir='..', # -- relative path for where to unzip the data, or None
    )
}

DATASET_DIR = 'PHENICX-Anechoic'

DATA = utils.LargeData('phenicx_anechoic_index.json')

DATASET_SECTIONS = {'doublebass':'strings','cello':'strings','clarinet':'woodwinds',
            'viola':'strings','violin':'strings','oboe':'woodwinds',
            'flute':'woodwinds','trumpet':'brass','bassoon':'woodwinds','horn':'brass'}

class Track(track.Track):
    """Phenicx-Anechoic Track class
    # -- YOU CAN AUTOMATICALLY GENERATE THIS DOCSTRING BY CALLING THE SCRIPT:
    # -- `scripts/print_track_docstring.py my_dataset`
    # -- note that you'll first need to have a test track (see "Adding tests to your dataset" below)

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/Example`

    """
    def __init__(self, track_id, data_home=None):
        if track_id not in DATA.index:
            raise ValueError(
                '{} is not a valid track ID in Example'.format(track_id))

        self.track_id = track_id

        if data_home is None:
            data_home = utils.get_default_dataset_path(DATASET_DIR)

        self._data_home = data_home
        self._track_paths = DATA.index[track_id]
        self._audio_track_paths = {k:v for k,v in sorted(self._track_paths.items()) if 'audio-' in k}
        self._score_track_paths = {k:v for k,v in sorted(self._track_paths.items()) if 'score-' in k}

        #### parse the keys for the list of instruments
        self.instruments = sorted([source.replace('score-','')
                for source in self._track_paths.keys() if 'score-' in source])

        #### get the corresponding sections for the instruments
        self.sections = sorted(list(set(section for instrument,section in DATASET_SECTIONS.items()
                            if instrument in self.instruments)))

        audio_path = [v[0] for k,v in self._track_paths.items() if 'audio-' in k][0]
        score_path = [v[0] for k,v in self._track_paths.items() if 'score-' in k][0]

        self.audio_path = os.path.dirname(os.path.join(self._data_home, audio_path))
        self.annotation_path = os.path.dirname(os.path.join(self._data_home, score_path))

        #### add sources to track
        self.sources = collections.OrderedDict()
        target_instruments = {instrument:[] for instrument in self.instruments}
        target_sections = {section:[] for section in self.sections}
        mix = []

        for i,(k,audio_source) in enumerate(self._audio_track_paths.items()):
            source_name = os.path.basename(audio_source[0]).split('.')[0]
            instrument = source_name.rstrip(string.digits)

            ####source
            source = Source(name=source_name, stem_id=i, path=os.path.join(self._data_home, audio_source[0]))
            self.sources[source_name]=source

            ####add to targets
            mix.append(source)
            target_instruments[instrument].append(source)
            target_sections[DATASET_SECTIONS[instrument]].append(source)

        #### build the input mix using the sources
        self.mix = Target(sources=mix,
                    name='mix',
                    instruments=self.instruments,
                    score_path=self.annotation_path)

        self.targets = collections.OrderedDict()
        ####build targets for instruments using the sources in target_instruments
        for instrument in self.instruments:
            self.targets[instrument] = Target(sources=target_instruments[instrument],
                        name=instrument,
                        instruments=[instrument],
                        score_path=self.annotation_path)
        ####build targets for sections using the sources in target_sections
        for section in self.sections:
            instruments = list(set([source.name.rstrip(string.digits) for source in target_sections[section]]))
            self.targets[section] = Target(sources=target_sections[section],
                        name=section,
                        instruments=instruments,
                        score_path=self.annotation_path)


    def get_score(self,target):
        """Get the score for a given target

        Args:
            target (str): name of the target e.g. violin

        Returns:
            (namedtuple, utils.EventData): the score in format 'start_times', 'end_times', 'event'

        """
        assert target in self.targets.keys(),'target {} is not in the list of targets {}'.format(target,self.targets)
        return self.targets[target].score

    def get_original_score(self,target):
        """Get the original score for a given target

        Args:
            target (str): name of the target e.g. violin

        Returns:
            (namedtuple, utils.EventData): the score in format 'start_times', 'end_times', 'event'

        """
        assert target in self.targets.keys(),'target {} is not in the list of targets {}'.format(target,self.targets)
        return self.targets[target].original_score

    def get_audio_mix(self):
        """Get the audio, sampling rate for the mix of sources

        Returns:
            (np.ndarray): the mono audio signal
            (float): The sample rate of the audio file

        """
        return self.mix.audio,self.mix.rate

    def get_audio_target(self,target):
        """Get the audio, sampling rate for a given target

        Args:
            target (str): name of the target e.g. violin

        Returns:
            (np.ndarray): the audio signal
            (float): The sample rate of the audio file

        """
        assert target in self.targets.keys(),'target {} is not in the list of targets {}'.format(target,self.targets)
        return self.targets[target].audio,self.targets[target].rate

    def get_audio_source(self,source):
        """Get the audio, sampling rate for a given source

        Args:
            source (str): name of the source e.g. violin1

        Returns:
            (np.ndarray): the audio signal
            (float): The sample rate of the audio file

        """
        assert source in self.sources.keys(),'source {} is not in the list of sources {}'.format(source,self.sources)
        return self.sources[source].audio,self.sources[source].rate


    def to_jams(self):
        """Jams: the track's data in jams format"""
        # score_data = [(self.targets[target].score, 'score-'+target) for target in self.targets]
        # original_score_data = [(self.targets[target].original_score, 'score-'+target) for target in self.targets]
        metadata = {}
        metadata['instruments'] = self.instruments
        metadata['sections'] = self.sections
        metadata['mix'] = self.mix
        metadata['sources'] = self.sources
        metadata['targets'] = self.targets
        audio_paths = [os.path.join(self.audio_path,source+'.wav') for source in self.sources.keys()]
        return jams_utils.jams_converter(
            audio_path=audio_paths[0],
            #event_data=score_data.extend(original_score_data),
            metadata=metadata,
        )

def load_audio(audio_path):
    """Load an audio file.

    Args:
        audio_path (str): path to audio file

    Returns:
        y (np.ndarray): the mono audio signal
        sr (float): The sample rate of the audio file

    """
    # -- for example, the code below. This should be dataset specific!
    # -- By default we load to mono
    # -- change this if it doesn't make sense for your dataset.
    if not os.path.exists(audio_path):
        raise IOError("audio_path {} does not exist".format(audio_path))
    return librosa.load(audio_path, sr=None, mono=True)

# -- the partial_download argument can be removed if `dataset.REMOTES` is missing/has only one value
# -- the force_overwrite argument can be removed if the dataset does not download anything
# -- (i.e. there is no `dataset.REMOTES`)
# -- the cleanup argument can be removed if the dataset has no tar or zip files in `dataset.REMOTES`.
def download(
    data_home=None, partial_download=None, force_overwrite=False, cleanup=True
):
    """Download the dataset.

    Args:
        data_home (str):
            Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
        force_overwrite (bool):
            Whether to overwrite the existing downloaded data
        partial_download (list):
            List indicating what to partially download. The list can include any of:
                * 'TODO_KEYS_OF_REMOTES' TODO ADD DESCRIPTION
            If `None`, all data is downloaded.
        cleanup (bool):
            Whether to delete the zip/tar file after extracting.

    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    download_utils.downloader(
        # -- everything will be downloaded & uncompressed inside `data_home`
        data_home,
        # -- by default all elements in REMOTES will be downloaded
        remotes=REMOTES,
        # -- we allow partial downloads of the datasets containing multiple remote files
        # -- this is done by specifying a list of keys in partial_download (when using the library)
        partial_download=partial_download,
        # -- if you need to give the user any instructions, such as how to download
        # -- a dataset which is not freely availalbe, put them here
        info_message=None,
        force_overwrite=force_overwrite,
        cleanup=cleanup,
    )


# -- keep this function exactly as it is
def validate(data_home=None, silence=False):
    """Validate if the stored dataset is a valid version

    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        missing_files (list): List of file paths that are in the dataset index
            but missing locally
        invalid_checksums (list): List of file paths that file exists in the dataset
            index but has a different checksum compare to the reference checksum
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    missing_files, invalid_checksums = utils.validator(
        DATA.index, data_home, silence=silence
    )
    return missing_files, invalid_checksums


# -- keep this function exactly as it is
def track_ids():
    """Return track ids

    Returns:
        (list): A list of track ids
    """
    return list(DATA.index.keys())


# -- keep this function as it is
def load(data_home=None):
    """Load  dataset
    Args:
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets`
    Returns:
        (dict): {`track_id`: track data}
    """
    if data_home is None:
        data_home = utils.get_default_dataset_path(DATASET_DIR)

    data = {}
    for key in DATA.index.keys():
        data[key] = Track(key, data_home=data_home)
    return data


# -- Write any necessary loader functions for loading the dataset's data
def load_score(score_paths):
    """
    Args:
        score_paths (str) or list[str]: list of txt score files containing start_time, end_time, note
    Returns:
        utils.EventData: score as EventData tuple
    """

    if isinstance(score_paths,str): score_paths=list(score_paths)
    assert isinstance(score_paths,list), "score_paths should be either string or list of strings"

    start_times = []
    end_times = []
    score = []
    for path in score_paths:
        if not os.path.exists(path):
            raise IOError("path {} does not exist".format(path))

        #### read start, end times
        times = np.loadtxt(path, delimiter=",",usecols=[0, 1], dtype=np.float)
        start_times.append(times[:,0])
        end_times.append(times[:,1])

        #### read notes as string
        with open(path) as f:
            content = f.readlines()
            sc = np.array([line.split(',')[2].strip('\n') for line in content])
            score.append(sc)

    start_times = np.concatenate(start_times)
    end_times = np.concatenate(end_times)
    score = np.concatenate(score)

    #sort on the start time
    ind = np.argsort(start_times, axis=0)
    start_times = np.take_along_axis(start_times, ind, axis=0)
    end_times = np.take_along_axis(end_times, ind, axis=0)
    score = np.take_along_axis(score, ind, axis=0)

    data = utils.EventData(start_times, end_times, score)
    return data


def cite():
    """Print the reference"""

    cite_data = """
=========== MLA ===========
Miron, Marius, et al. "Score-informed source separation for multichannel orchestral recordings." Journal of Electrical and Computer Engineering 2016 (2016).

Pätynen, Jukka, Ville Pulkki, and Tapio Lokki. "Anechoic recording system for symphony orchestra." Acta Acustica united with Acustica 94.6 (2008): 856-865.
========== Bibtex ==========
@article{miron2016score,
  title={Score-informed source separation for multichannel orchestral recordings},
  author={Miron, Marius and Carabias-Orti, Julio J and Bosch, Juan J and G{\'o}mez, Emilia and Janer, Jordi},
  journal={Journal of Electrical and Computer Engineering},
  volume={2016},
  year={2016},
  publisher={Hindawi}
}

@article{patynen2008anechoic,
  title={Anechoic recording system for symphony orchestra},
  author={P{\"a}tynen, Jukka and Pulkki, Ville and Lokki, Tapio},
  journal={Acta Acustica united with Acustica},
  volume={94},
  number={6},
  pages={856--865},
  year={2008},
  publisher={S. Hirzel Verlag}
}

"""
    print(cite_data)



##########################################
#### derived from musdb multi-track code
#### distributed under MIT license
##########################################

class Source(object):
    """An audio Target which is a linear mixture of several sources

    Args:
        name (str): Name of this source
        stem_id (int): stem/substream ID is set here.
        path (str): Absolute path to audio file
        gain (float): Mixing weight for this source
    """
    def __init__(
        self,
        name=None,  # has its own name
        path=None,  # might have its own path
        stem_id=None,  # might have its own stem_id
        gain=1.0,
        *args,
        **kwargs
    ):
        self.name = name
        self.path = path
        self.stem_id = stem_id
        self.gain = gain
        self._audio = None

    def __repr__(self):
        return self.path

    @property
    def audio(self):
        # return cached audio if explicitly set by setter
        if self._audio is not None:
            return self._audio
        # read from disk to save RAM otherwise
        else:
            audio, self._rate = load_audio(self.path)
            self.shape = audio.shape
            return audio

    @audio.setter
    def audio(self, array):
        self._audio = array

    @property
    def rate(self):
        return self._rate

    def __eq__(self, other):
        """ tests if two sources are equal
        """
        if not isinstance(other, Source):
            return False
        else:
            return self.name==other.name and self.gain==other.gain and os.path.basename(self.path)==os.path.basename(other.path)


# Target from musdb DB mixed from several sources
class Target(object):
    """
    An audio Target which is a linear mixture of several sources/targets
    Attributes

    Args:
        sources (list[Source/Target]): list of ``Source`` objects for this ``Target``
    """
    def __init__(
        self,
        sources, # list of Source objects
        instruments, # list of str (instruments)
        score_path, # paths to score/annotation files
        name=None,  # has its own name
    ):
        assert isinstance(sources,list),"sources should be a list of Source objects"
        assert isinstance(instruments,list),"instruments should be a list of str representing instruments"
        print(score_path)
        assert os.path.isdir(score_path),"score_path does not exist"
        self.sources = sources
        self.name = name
        self.score_path = score_path
        self.instruments = sorted(instruments)

    @property
    def audio(self):
        """array_like: [shape=(num_samples)]
        mixes audio for targets on the fly
        """
        for i,source in enumerate(self.sources):
            audio = source.audio
            sr = source.rate
            if audio is not None:
                if i==0:
                    mix = source.gain * audio
                    self._rate = sr
                else:
                    assert sr==self.rate,"the sampling rate is different for two sources of the same target"
                    if len(audio)>len(mix):
                        prev_len = len(mix)
                        mix = np.resize(mix,audio.shape)
                        mix[prev_len:] = 0.
                        mix += source.gain * audio
                    elif len(audio)<len(mix):
                        mix[:len(audio)] += source.gain * audio
                    else:
                        mix += source.gain * audio
        return mix

    @property
    def rate(self):
        return self._rate

    @utils.cached_property
    def score(self):
        """ returns the score
        """
        if not os.path.isdir(self.score_path):
            raise IOError("path {} does not exist".format(self.score_path))
        score_paths = [os.path.join(self.score_path,instrument+'.txt')
                                    for instrument in self.instruments]
        return load_score(score_paths)

    @utils.cached_property
    def original_score(self):
        """ returns the original score
        """
        if not os.path.isdir(self.score_path):
            raise IOError("path {} does not exist".format(self.score_path))
        score_paths = [os.path.join(self.score_path,instrument+'_o.txt')
                                    for instrument in self.instruments]
        return load_score(score_paths)

    def __repr__(self):
        parts = []
        for source in self.sources:
            parts.append(source.name)
        return '+'.join(parts)

    def __eq__(self, other):
        """ tests if two targets are equal
        """
        if not isinstance(other, Target):
            print('not the same type')
            return False
        else:
            if self.name!=other.name:
                print('names not equal')
                return False
            if self.instruments!=other.instruments:
                print('instruments not equal')
                return False
            for s1,s2 in zip(self.sources,other.sources):
                if s1!=s2: return False
            return True
