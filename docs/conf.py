"""Sphinx configuration for the fastwindow documentation site.

The API reference is generated from ``fastwindow/__init__.pyi``: the stub
file carries the full numpy-style reference docstrings and exact
signatures, and it is loaded here as a real Python module so autodoc can
consume it WITHOUT needing the compiled extension at docs-build time.
"""

import importlib.machinery
import importlib.util
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

_pyi = os.path.join(ROOT, "fastwindow", "__init__.pyi")
_loader = importlib.machinery.SourceFileLoader("fastwindow_api", _pyi)
_spec = importlib.util.spec_from_loader("fastwindow_api", _loader)
_mod = importlib.util.module_from_spec(_spec)
_loader.exec_module(_mod)
sys.modules["fastwindow_api"] = _mod

with open(os.path.join(ROOT, "fastwindow", "__init__.py")) as fh:
    release = re.search(r'__version__ = "([^"]+)"', fh.read()).group(1)

project = "fastwindow"
author = "bagol1000"
copyright = "2026, bagol1000"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_member_order = "bysource"

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = f"fastwindow {release}"
