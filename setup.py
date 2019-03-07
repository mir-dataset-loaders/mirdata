""" Setup script for mir_datasets. """
from setuptools import setup

import imp

version = imp.load_source('mir_datasets.version', 'mir_datasets/version.py')

if __name__ == "__main__":
    setup(
        name='mir_datasets',
        version=version.version,
        description='Common loaders for MIR datasets.',
        author='Rachel Bittner',
        author_email='rachel.bittner@gmail.com',
        url='https://github.com/rabitt/mir_datasets',
        download_url='http://github.com/rabitt/mir_datasets/releases',
        packages=['mir_datsets'],
        package_data={'mir_datasets': []},
        long_description="""Common loaders for MIR datasets.""",
        keywords='mir dataset loader audio',
        license='BSD-3-Clause',
        install_requires=[
        ],
        extras_require={
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
            ],
            'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
        }
    )
