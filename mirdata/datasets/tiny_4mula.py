"""4MuLA-tiny Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    This dataset is a tiny version extracted by 4MuLA dataset:

    .. code-block:: latex
        "4MuLA: A Multitask, Multimodal, and Multilingual Dataset of Music Lyrics and Audio Features."
        by Angelo Cesar Mendes da Silva, Diego Furtado Silva, and Ricardo Marcondes Marcacini.
        In Proceedings of the Brazilian Symposium on Multimedia and the Web (WebMedia '20). 2020.
        DOI:https://doi.org/10.1145/3428658.3431089


    The dataset consists of 1988 musics represents by melspectrogram and metadata. It
    contains 27 genres and 93 artists. The audio file is not available. All attributes are:
        [
         'music_id', 'music_name', 'music_lang', 'music_lyrics', 'art_id',
         'art_name', 'art_rank', 'main_genre', 'related_genre', 'related_art',
         'related_music', 'musicnn_tags', 'melspectrogram'
        ]

"""
import os
from mirdata.download_utils import RemoteFileMetadata, downloader
from mirdata import core, jams_utils
from pandas import read_csv, read_parquet
from numpy import load, ndarray

BIBTEX = """@inproceedings{10.1145/3428658.3431089,
    author = {da Silva, Angelo Cesar Mendes and Silva, Diego Furtado and Marcacini, Ricardo Marcondes},
    title = {4MuLA: A Multitask, Multimodal, and Multilingual Dataset of Music Lyrics and Audio Features},
    year = {2020},
    isbn = {9781450381963},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3428658.3431089},
    doi = {10.1145/3428658.3431089},    
    booktitle = {Proceedings of the Brazilian Symposium on Multimedia and the Web},
    pages = {145–148},
    numpages = {4},
    keywords = {latin musical dataset, music dataset, multimodal musical dataset},
    location = {S\~{a}o Lu\'{\i}s, Brazil},
    series = {WebMedia '20}
}
"""

REMOTES = {
    "dataset": RemoteFileMetadata(
        filename="4mula_tiny.parquet",
        url="https://zenodo.org/record/4585498/files/4mula_tiny.parquet?download=1",
        checksum="b1caba387baa22a7b16893702a3214f7",
        destination_dir=".",
    )
}

LICENSE_INFO = "Creative Commons Attribution 4.0 International."

DOWNLOAD_INFO = """
    This loader is designed to load the melspectrogram and metadata, as it is available for download.
    Unfortunately the audio files of the 4MULA dataset are not available
    for download. You can more details and follow the updates in our repository: 
    ==> https://github.com/4mulaDataset/4mula    
"""

DEFAULT_COLUMNS = ['music_id', 'music_name', 'music_lang', 'music_lyrics', 'art_id', 'art_name',
                   'art_rank', 'main_genre', 'related_genre', 'related_art', 'related_music',
                   'musicnn_tags', 'melspectrogram']


class Track(core.Track):
    """tiny_4mula track class

    Args:
        track_id (str): track id of the track
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/tiny_4mula`

    Attributes:
        music_id (str): track id
        music_name (str): music name
        music_lang (str): music language
        music_lyrics (str): lyrics
        art_id (str): artist id
        art_name (str): artist name
        art_rank (int): artist rank
        main_genre (str): artist main genre
        related_genre (list): artist related genre
        related_art (list): others artist related to artist
        related_music (list): others music related to track
        musicnn_tags (list): top-3 tags got by Musicnn
        melspectrogram (numpy.ndarray): melspectrogram extracted by librosa


    Cached Properties:
        melspectrogram: melspectrogram in npy file
        annotation: csv file with annotated metadata (all attributes, except melspectrogram)

    """

    def __init__(
            self,
            track_id,
            data_home,
            dataset_name,
            index,
            metadata,
    ):
        super().__init__(
            track_id,
            data_home,
            dataset_name,
            index,
            metadata,
        )

        self.annotation_path = os.path.join(self._data_home, self._track_paths["annotation"][0])
        self.melspectrogram_path = os.path.join(self._data_home, self._track_paths["melspectrogram"][0])

    @property
    def music_id(self):
        return self._track_metadata.get("music_id")[0]

    @property
    def music_name(self):
        return self._track_metadata.get("music_name")[0]

    @property
    def music_lang(self):
        return self._track_metadata.get("music_lang")[0]

    @property
    def music_lyrics(self):
        return self._track_metadata.get("music_lyrics")[0]

    @property
    def art_id(self):
        return self._track_metadata.get("art_id")[0]

    @property
    def art_name(self):
        return self._track_metadata.get("art_name")[0]

    @property
    def art_rank(self):
        return self._track_metadata.get("art_rank")[0]

    @property
    def main_genre(self):
        return self._track_metadata.get("main_genre")[0]

    @property
    def related_art(self):
        return eval(self._track_metadata.get("related_art")[0])

    @property
    def related_genre(self):
        return eval(self._track_metadata.get("related_genre")[0])

    @property
    def related_music(self):
        return eval(self._track_metadata.get("related_music")[0])

    @property
    def musicnn_tags(self):
        return eval(self._track_metadata.get("musicnn_tags")[0])

    @property
    def _track_metadata(self):
        metadata = read_csv(self.annotation_path, sep="\t")
        if not metadata.empty:
            return metadata
        return None

    @property
    def melspectrogram(self):
        return load(self.melspectrogram_path, allow_pickle=True)

    def to_jams(self):
        """Get the track's data in jams format

        Returns:
            jams.JAMS: the track's data in jams format

        """
        return jams_utils.jams_converter(
            spectrogram_path=self.melspectrogram_path,
            metadata={
                'art_id': self.art_id,
                'art_name': self.art_name,
                'art_rank': self.art_rank,
                'main_genre': self.main_genre,
                'music_id': self.music_id,
                'music_lang': self.music_lang,
                'music_lyrics': self.music_lyrics,
                'music_name': self.music_name,
                'musicnn_tags': self.musicnn_tags,
                'related_art': self.related_art,
                'related_genre': self.related_genre,
                'related_music': self.related_music,
                'duration': 30  # approximation
            }
        )


def load_melspectrogram(fhandle: str) -> ndarray:
    """Load a tiny_4mula dataset melspectrogram.

    Args:
        fhandle (str): path pointing to the dataset file

    Returns:
        numpy.ndarray: melspectrogram

    """
    print(str(fhandle))
    if not os.path.isfile(str(fhandle)):
        raise FileNotFoundError("Dataset not found. Did you run .download()?")
    df = read_parquet(fhandle, columns=["melspectrogram"])
    return df['melspectrogram'].values


def load_by_columns(fhandle: str, columns: list = DEFAULT_COLUMNS):
    """Load a tiny_4mula dataset filtering by columns.

    Args:
        fhandle (str): path pointing to the dataset file
        columns (list): attributes to filter in dataset read

    Returns:
        pandas.DataFrame: tiny_4mula dataset filtered by columns

    """
    if not os.path.isfile(str(fhandle)):
        raise FileNotFoundError("Dataset not found. Did you run .download()?")
    return read_parquet(fhandle, columns=columns)


@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """
    The tiny version of 4MuLA dataset
    """

    def __init__(self, data_home=None):
        super().__init__(
            data_home,
            name="tiny_4mula",
            track_class=Track,
            bibtex=BIBTEX,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    @core.cached_property
    def metadata(self):
        metadata_path = os.path.join(self.data_home, "4mula_tiny.parquet")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("Metadata not found. Did you run .download()?")
        columns = ['music_id', 'music_name', 'music_lang', 'music_lyrics', 'art_id', 'art_name',
                   'art_rank', 'main_genre', 'related_genre', 'related_art', 'related_music',
                   'musicnn_tags']
        return read_parquet(metadata_path, columns=columns)

    @core.copy_docs(load_melspectrogram)
    def load_spectrogram(self, *args, **kwargs):
        return load_melspectrogram(*args, **kwargs)

    def download(self, partial_download=None, force_overwrite=False, cleanup=False):
        """Download the dataset

        Args:
            partial_download (list or None):
                A list of keys of remotes to partially download.
                If None, all data is downloaded
            force_overwrite (bool):
                If True, existing files are overwritten by the downloaded files.
            cleanup (bool):
                Whether to delete any zip/tar files after extracting.

        Raises:
            ValueError: if invalid keys are passed to partial_download
            IOError: if a downloaded file's checksum is different from expected

        """
        downloader(
            self.data_home,
            remotes=REMOTES,
            partial_download=partial_download,
            force_overwrite=force_overwrite,
            cleanup=cleanup,
        )
