..
  :vim: set softtabstop=2 :
  
  Short documentation for the nudie project. 

Outline
=======

- What is it?
  - a new set of analysis scripts
  - works using python 3
  - goals
  - only works as long as data taking layout doesn't change
- How does it work?
  - conventions for data storage
  - workflow for analysis (should include description of data taking)
  - some tricks used
- How do I use it? 
  - look at example script and try to write your own

Code documentation
==================

Working with spectra
--------------------

.. py:function:: freq_to_wavelen(axis, data, [ret_dwl = False], [ax = -1])

