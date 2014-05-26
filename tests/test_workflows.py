'''more integrated tests that use many functions'''

import pytest
import tempfile
import itertools as it
import numpy as np
import random

import pathlib
import nudie

@pytest.fixture(scope='function')
def datadir():
    '''creates a data directory with some files'''
    t2 = 1
    tables = 10
    loops = 4
    batches = 1
    jobname = 'testjob'

    xdim = 1340
    ydim = 1
    frames = 1000
    
    phases = 6
    repeat = 1
    waveforms = 40

    waveform_list = list(range(waveforms))

    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d)
        for b in range(batches):
            bp = path.mkdir(jobname + '-{:2d}'.format(b))

            for t2,tb,i in it.product(range(t2), range(tables), range(loops)):

                template = jobname + '{:2d}-{:2d}-{:2d}'.format(t2,tb,i)

                # make datafile
                header = nudie.utils.winspec.Header()
                header.xdim, header.ydim = xdim, ydim
                header.NumFrames = frames
                header.datatype = 3 # uint16 datatype
                data = np.zeros((frames, ydim, xdim), dtype='uint16')

                data[0:3, :, :] = (1<<16) - 1 # saturated frames

                with bp.join(template + '.spe').open('wb') as f:
                    f.write(header)
                    f.write(data)
                
                analog_len = (frames - random.randint(1,5),)
 
                with bp.join(template + '-analog1.txt') as f:

                    data = np.zeros(analog_len, dtype=float)
                    data[3+random.sample(waveform_list)::waveforms*repeat*phases] = 5.0
                    np.savetxt(f, data, delimiter=',')

                with bp.join(template + '-analog2.txt') as f:
                    data = np.zeros(analog_len, dtype=float)
                    # FIXME: WIP

def test_6phase_on_off():
    pass
