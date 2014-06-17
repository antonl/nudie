import pytest
import nudie
import numpy as np
import itertools as it
import pdb


@pytest.mark.parametrize("period, offset", [
    (6, 2), (4, 1), (12, 4), (5, 2), (5, 0),
    ])
def test_several_waveforms_one_phase_offset(period, offset):
    num_phases = 1
    nframes = 1000
    duty_cycle = 0.5
    table_start = np.array([period*i + offset for i in range(nframes//period)])

    shutter_info = {'last open idx': slice(None, int(nframes*duty_cycle)),
            'first closed idx' : slice(int(nframes*duty_cycle)+5, None)}

    tags = nudie.tag_phases(table_start, period, range(num_phases), \
            nframes, shutter_info=shutter_info)

    assert len(tags) == 1, 'waveform repeat detected incorrectly'
    assert len(tags[0]) == period, 'nwaveforms detected incorrectly'
    assert len(tags[0][0]) == num_phases, 'num_phases detected incorrectly'

    assert tags[0][0][0]['shutter open'][0] == offset
    # first repeat, offset'th waveform, first phase, shutter open, and first
    # index
    assert tags[0][-offset % period][0]['shutter open'][0] == 0, \
            'incorrect first waveform'

@pytest.mark.parametrize('inp', [(1,), (2,), (4,)])
def test_nrepeat(inp):
    nphases = 6
    nwaveforms = 10 
    nframes = 1000
    duty_cycle = 0.5
    reps = inp[0]
    offset = 2

    period = nphases*nwaveforms*reps

    table_start = np.array([period*i + offset for i in range(nframes//period)])
    shutter_info = {'last open idx': slice(None, int(nframes*duty_cycle)),
            'first closed idx' : slice(int(nframes*duty_cycle)+5, None)}

    tags = nudie.tag_phases(table_start, period, range(nphases), \
            nframes, waveform_repeat=reps, shutter_info=shutter_info)

    assert len(tags) == reps, 'incorrect detected reps'

def test_exact_period():
    # Fixed bug where an assertion would fail if table_start_detect[0]
    # indicated that the first camera frame was the first phase in the 
    # waveform

    period = 4
    num_phases = 4
    nframes = 100
    duty_cycle = 0.5
    table_start = np.array([4*i for i in range(nframes//period)], dtype=int)

    shutter_info = {'last open idx': slice(None, int(nframes*duty_cycle)),
            'first closed idx' : slice(int(nframes*duty_cycle)+5, None)}

    tags = nudie.tag_phases(table_start, period, range(num_phases), \
            nframes, shutter_info=shutter_info)

    # check that the first phase of the only waveform with the shutter open 
    # starts at index 0
    assert tags[0][0][0]['shutter open'][0] == 0 

@pytest.mark.parametrize("offset, repeat", [
    (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
    (5, 1), (6, 1), (7,1)])
def test_general_case(offset, repeat):
    ''' tests whether many waveforms, several phase case works'''
    duty_cycle = 0.5
    transition_width = 3
    nframes = 100 
    waveform_repeat = repeat
    nwaveforms = 3 
    tags = ['one']
    nphases = len(tags) 

    data = np.zeros((nframes, 4), dtype=object)
    analog1 = np.zeros((nframes,), dtype=float)    

    states = it.cycle([(rep, phase, waveform) \
            for waveform in range(nwaveforms) \
            for phase in tags \
            for rep in range(waveform_repeat)])

    #states, tmp = it.tee(states, 2)
    #test = list(it.islice(tmp, offset, 100))

    # set shutter
    last_on = int(nframes*duty_cycle)

    shutter_it = it.chain(it.repeat('shutter open', last_on), it.repeat(None,
        transition_width), it.repeat('shutter closed'))
    
    for i, (rep, phase, waveform), shut in \
            zip(range(nframes), it.islice(states, offset, None), shutter_it):
        data[i] = (rep, phase, waveform, shut)
        
        if waveform == 0 and phase == tags[0]: analog1[i] = 2.5

    start, period = nudie.detect_table_start(analog1, waveform_repeat)
    
    shutter_info = {'last open idx': slice(None, last_on+1),
            'first closed idx' : slice(last_on+transition_width, None)}

    tagged = nudie.tag_phases(start, period, tags, nframes, \
            waveform_repeat=waveform_repeat, shutter_info=shutter_info)

    for i, (r,p,w,s) in enumerate(data):
        if s is None: continue # skip transition frames
        assert i in tagged[r][w][p][s]

