from setuptools import setup, find_packages
setup(
    name = "nudiepy",
    version = "0.1",
    packages = find_packages(),
    #scripts = ['say_hello.py'],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires = ['pathlib'],

    # metadata for upload to PyPI
    author = "Anton Loukianov",
    author_email = "aloukian@umich.edu",
    description = "This is a set of tools that I use to analyze 2D spectra. 
    The goal is to have a unified, tested, and version-controlled set of analysis 
    scripts that \"just work.\"",
    license = "GPL",
    keywords = "ultrafast spectroscopy 2D fourier transform",
    url = "http://umich.edu/~aloukian/nudiepy",   # project home page, if any

    # could also include long_description, download_url, classifiers, etc.
)
