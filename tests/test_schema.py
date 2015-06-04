from __future__ import division
import pytest
import numpy as np
import itertools as it

import nudie

nudie.show_errors(nudie.logging.DEBUG)

def test_BoolStr():
    for s in ['True', '1', 'tRue', 'YES', 'ON', 'yEs']:
        assert nudie.schema.BoolStr(s) == True
    for s in ['false', '0', 'FaLSE', 'OFF', 'ofF', 'NO', 'no']:
        assert nudie.schema.BoolStr(s) == False

def test_parse_config_basic(tmpdir):

    p = tmpdir.join('test_cfg.ini')

    p.write(\
    '''
    [pump probe]
    jobname = asldkfja
    batch = 10
    when = 09123
    wavelengths = alkdfja
    analysis path = alksdjfalskdjf
    ''')

    with pytest.raises(ValueError):
        nudie.parse_config(p.strpath, which='asldkajslkdj')

    with pytest.raises(ValueError):
        nudie.parse_config(p.strpath, which='2d')

    with pytest.raises(ValueError):
        nudie.parse_config(p.strpath, which='phasing')


    nudie.parse_config(p.strpath, which='pump probe')




