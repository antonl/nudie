# Nudie: a set of analysis scripts for the new 2D side of the Ogilvie Group

This is a set of tools that I use to analyze 2D spectra. The goal is to have a
unified, tested, and version-controlled set of analysis scripts that "just
work."

## Requirements

- Python 2.7 or 3.4
- pathlib module (included in 3.4)

## Global variables

Attempts have been made to avoid nonlocal variables, but sometimes this is
inconvenient. Here is a list of global variables that are used in the scripts:

- `mount_point`: a `pathlib.Path` object that represents the location of the
  data drive as chosen by the OS you are running.

