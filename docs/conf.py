# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "mirdata"
copyright = "2019-2020, mirdata development team."
author = "The mirdata development team"


import importlib

mirdata_version = importlib.import_module("mirdata.version")

# The short X.Y version.
version = mirdata_version.short_version
# The full version, including alpha/beta/rc tags.
release = mirdata_version.version
# Show only copyright
show_authors = False

# -- Mock dependencies -------------------------------------------------------
autodoc_mock_imports = [
    "librosa",
    "numpy",
    "pretty_midi",
    "DALI",
    "music21",
    "h5py",
    "yaml",
    "scipy",
    "smart_open",
    "openpyxl",
    "pandas",
]

# # -- General configuration ---------------------------------------------------

# # Add any Sphinx extension module names here, as strings. They can be
# # extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# # ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_togglebutton",
    "sphinx.ext.extlinks",
    "sphinx_copybutton",
]

# To shorten links of licenses and add to table
extlinks = {
    "acousticbrainz": ("https://zenodo.org/record/2554044#.X_ivJ-n7RUI%s", "Custom%s"),
    "cante": ("https://zenodo.org/record/1324183#.X_nq7-n7RUI%s", "Custom%s"),
    "ikala": ("http://mac.citi.sinica.edu.tw/ikala/%s", "Custom%s"),
    "rwc": ("https://staff.aist.go.jp/m.goto/RWC-MDB/%s", "Custom%s"),
    "tonas": ("https://www.upf.edu/web/mtg/tonas/%s", "Custom%s"),
}


intersphinx_mapping = {
    "np": ("https://numpy.org/doc/stable/", None),
    "mir_eval": ("https://mir-eval.readthedocs.io/latest/", None),
    "pretty_midi": ("https://craffel.github.io/pretty-midi/", None),
}

# Napoleon settings
# https://github.com/sphinx-contrib/napoleon/issues/2
napoleon_custom_sections = [
    ("Cached Properties", "Other Parameters")
]  # todo - when above issue is closed, update to say "cached properties"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_type_aliases = None
napoleon_attr_annotations = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "source/example.rst",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

html_logo = "img/mirdata.png"
