""" Setup script for mir_datasets. """
from setuptools import setup, find_packages

# import imp

# version = imp.load_source(
#     'mirdata.version', 'mirdata/version.py')

if __name__ == '__main__':
    setup(
        name='mirdata',
        version='0.0.1',  # version.version,
        description='Common loaders for MIR datasets.',
        url='https://github.com/mir-dataset-loaders/mirdata',
        packages=find_packages(exclude=['test', '*.test', '*.test.*']),
        download_url='http://github.com/mir-dataset-loaders/mirdata/releases',
        package_data={'mirdata': ['indexes/*']},
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
                'testcontainers'
            ],
            'docs': [
                'sphinx',
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
        }
    )
