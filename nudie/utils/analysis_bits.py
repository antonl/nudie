'''
contains emperical voodoo that seems to make our camera and labview software
work happily together. 
'''
import logging
log = logging.getLogger('nudie.analysis_bits')
import itertools as it
import numpy as np
from pathlib import Path

def repeatn(iter, n=1):
    '''repeat each item in an iterator n times'''
    for item in iter:
        repeater = it.repeat(item, n)
        for x in repeater:
            yield x

def cleanup_analogtxt(file):
    '''remove fill values in the analog txt files'''
    fillval = -1
    try:
        file = Path(file)
        with file.open(mode='rb') as f:
            arr = np.genfromtxt(f, dtype=float)

            assert len(arr.shape) == 1, "should be single column file"
            if not np.all(np.not_equal(arr, np.nan)):
                s = '`{!s}` contains invalid data. NANs detected!'.format(file)
                log.error(s)
                raise RuntimeError(s)

            farr = arr[arr > fillval]
            return farr, farr.shape[0]
    except Exception as e:
        log.error('could not read file `{!s}`'.format(file))
        raise e

def detect_table_start(array, repeat=1):
    '''detects peaks output when the Dazzler table starts at the begining by
    looking at the discrete difference of the array.
    
    '''
    diff = 2 # at least 2 Volt difference
    darr = np.diff(array)
    lohi, hilo = np.argwhere(darr > 2), np.argwhere(darr < -2)

    assert len(lohi) == len(hilo), 'spike showed up as the first or last signal?'
    if not np.allclose(hilo-lohi, repeat):
        s = 'detected incorrect repeat or a break in periodicity ' +\
            'of the table. Check that the DAQ card cable is connected ' + \
            'and dazzler synchronization is working correctly. ' + \
            'Alternatively, check the waveform repeat setting. ' + \
            'Repeat is set to {:d}.'.format(repeat)
        log.error(s)
        raise RuntimeError(s)
    # diff returns forward difference. Add one for actual peak position 
    return np.squeeze(lohi)+1 

def tag_phases(table_start_detect, waveform_range, waveform_repeat=1):
    '''tag camera frames based on the number of waveforms and waveform repeat'''
    pass

