# Configuration file for the Sphinx documentation builder.

import os
import sys

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'timewizard'
copyright = '2023, Jonah Pearl'
author = 'Jonah Pearl'
release = 'v0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "autodocsumm",
    "myst_nb",
]

myst_enable_extensions = [
    "colon_fence",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
    ".ipynb": "myst-nb"
}
jupyter_execute_notebooks = "off"

autodoc_default_options = {
    "autosummary": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_show_sourcelink = True

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "jonahpearl",  # Username
    "github_repo": "timewizard",  # Repo name
    "github_version": "master",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}


autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# trying to fix rtd...
sys.path.insert(0, os.path.abspath('../src'))
