""" Setup script for mir_datasets. """
from setuptools import setup

import imp

version = imp.load_source(
    'mir_dataset_loaders.version', 'mir_dataset_loaders/version.py')

if __name__ == "__main__":
    setup(
        name='mir_dataset_loaders',
        version=version.version,
        description='Common loaders for MIR datasets.',
        url='https://github.com/mir-dataset-loaders/mir-dataset-loaders',
        download_url='http://github.com/rabitt/mir-dataset-loaders/releases',
        packages=['mir_dataset_loaders'],
        package_data={'mir_dataset_loaders': []},
        long_description="""Common loaders for MIR datasets.""",
        keywords='mir dataset loader audio',
        license='BSD-3-Clause',
        install_requires=[
            'tqdm',
            'librosa',
            'numpy',
        ],
        extras_require={
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
            ],
            'docs': [
                'sphinx',
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
        }
    )
