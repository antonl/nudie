from setuptools import setup, find_packages
import re
import ast

_version_re = re.compile(r'version\s+=\s+(.*)')

with open('nudie/version.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

setup(
    name = "nudiepy",
    version = version,
    packages = find_packages(),
    #scripts = ['say_hello.py'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires = ['numpy'],

    # metadata for upload to PyPI
    author = "Anton Loukianov",
    author_email = "aloukian@umich.edu",
    description = "This is a set of tools that I use to analyze 2D spectra. The goal is to have a unified, tested, and version-controlled set of analysis scripts that \"just work.\"",
    license = "GPL",
    keywords = "ultrafast spectroscopy 2D fourier transform",
    url = "http://umich.edu/~aloukian/nudiepy",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
