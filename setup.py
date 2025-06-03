""" Setup script for mir_datasets. """
from importlib.machinery import SourceFileLoader
from setuptools import setup, find_packages

version_sfl = SourceFileLoader("mirdata.version", "mirdata/version.py")
version_module = version_sfl.load_module() 

if __name__ == "__main__":
    setup(
        version=version_module.version,
                packages=find_packages(exclude=["test", "*.test", "*.test.*","tests", "tests.*", "*.tests.*", "tests/__pycache__", "tests/__pycache__/*"]),

    )