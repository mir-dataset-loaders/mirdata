#!/usr/bin/env python

"""
Command-line interface for mirdata

e.g.:
>>> python -m mirdata --list
>>> python -m mirdata 'orchset' --no-validate
"""

from pathlib import Path

from mirdata.core import Dataset
from . import list_datasets, initialize


def main(dataset, list=False, destination=None, force=False, version=None, **kwargs):
    if list:
        _list_datasets_to_console(destination)
        return

    print(f"Preparing download of {dataset}")
    dataset = initialize(dataset, destination)
    dataset.download(force_overwrite=force)
    dataset.validate()


def _list_datasets_to_console(downloaded:Path=None):
    if downloaded is not None and downloaded.exists():
        print("Downloaded datasets")
        print("-------------------")
        subdirectories = filter(Path.is_dir, downloaded.iterdir())
        print("\n".join(map(str, subdirectories)))
        print("\n\nAvailable datasets")
        print("--------------------")

    print("\n".join(list_datasets()))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', nargs='?',
                        help="name of the dataset to download/validate")
    parser.add_argument('--data-home', '-d', default=Path(Dataset._default_dir()), type=Path,
                        help="target directory the dataset is be downloaded")
    parser.add_argument('--list', '-l', action='store_true',
                        help="list all available datasets")
    parser.add_argument('--no-validate', dest='validate', action='store_false',
                        help='skip dataset validation')
    parser.add_argument('--force', '-f',
                        help="overwrite dataset if it exists")
    parser.add_argument('--version', '-v',
                        help="dataset version")
    parser.add_argument('--citation', '-c',
                        help="Only print the citation, don't download")
    args = parser.parse_args()

    # argument validation
    if args.dataset is None and args.list is False:
        parser.error("the following arguments are required: dataset")

    main(**vars(args))
