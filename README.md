# mirdata
common loaders for Music Information Retrieval (MIR) datasets. Find the API documentation [here](https://mirdata.readthedocs.io/en/latest/).

[![CircleCI](https://circleci.com/gh/mir-dataset-loaders/mirdata.svg?style=svg)](https://circleci.com/gh/mir-dataset-loaders/mirdata)
[![codecov](https://codecov.io/gh/mir-dataset-loaders/mirdata/branch/master/graph/badge.svg)](https://codecov.io/gh/mir-dataset-loaders/mirdata)
[![Documentation Status](https://readthedocs.org/projects/mirdata/badge/?version=latest)](https://mirdata.readthedocs.io/en/latest/?badge=latest)
![GitHub](https://img.shields.io/github/license/mir-dataset-loaders/mirdata.svg)


This library provides tools for working with common MIR datasets, including tools for:
* downloading datasets to a common location and format
* validating that the files for a dataset are all present 
* loading annotation files to a common format, consistent with the format required by [mir_eval](https://github.com/craffel/mir_eval)
* parsing track level metadata for detailed evaluations


### Installation

To install, simply run:

```python
pip install mirdata
```

### Quick example
```python
import mirdata
import random

orchset = mirdata.Dataset('orchset')
orchset.download()  # download the dataset
orchset.validate()  # validate that all the expected files are there

example_track = orchset.choice_track()  # choose a random example track
print(example_track)  # see the available data
```
See the [documentation](https://mirdata.readthedocs.io/en/latest/) for more examples and the API reference.


### Currently supported datasets


Supported datasets include [AcousticBrainz](https://zenodo.org/record/2553414#.X8jTgulKhhE), [DALI](https://github.com/gabolsgabs/DALI), [Guitarset](http://github.com/marl/guitarset/), [MAESTRO](https://magenta.tensorflow.org/datasets/maestro), [TinySOL](https://www.orch-idea.org/), among many others.

For the **complete list** of supported datasets, see the [documentation](https://mirdata.readthedocs.io/en/latest/source/quick_reference.html)


### Citing


There are two ways of citing mirdata:

If you are using the library for your work, please cite the version you used as indexed at Zenodo:

DOI

If you refer to mirdata's design principles, motivation etc., please cite the following [paper](https://magdalenafuentes.github.io/publications/2019_ISMIR_mirdata.pdf):

```
"mirdata: Software for Reproducible Usage of Datasets"
Rachel M. Bittner, Magdalena Fuentes, David Rubinstein, Andreas Jansson, Keunwoo Choi, and Thor Kell
in International Society for Music Information Retrieval (ISMIR) Conference, 2019
```

```
@inproceedings{
  bittner_fuentes_2019,
  title={mirdata: Software for Reproducible Usage of Datasets},
  author={Bittner, Rachel M and Fuentes, Magdalena and Rubinstein, David and Jansson, Andreas and Choi, Keunwoo and Kell, Thor},
  booktitle={International Society for Music Information Retrieval (ISMIR) Conference},
  year={2019}
}
```

### Contributing a new dataset loader

We welcome contributions to this library, especially new datasets. Please see [contributing](https://mirdata.readthedocs.io/en/latest/source/contributing.html) for guidelines.
