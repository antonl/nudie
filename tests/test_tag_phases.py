import pytest
import nudie
import numpy as np
import itertools as it
import pdb

def test_general_case():
    ''' tests whether many waveforms, several phase case works'''
    duty_cycle = 0.5
    transition_width = 3
    N = 100 
    offset = 3
    repeat = 2
    num_waveforms = 6 
    num_phases = 3 
    data = np.zeros((N, 4), dtype=object)

    for r,w,p in it.product(range(repeat), range(num_waveforms), 
            range(num_phases)):
        data[r + p*repeat + w*repeat*num_phases \
                ::repeat*num_phases*num_waveforms] = (r, w, p, 0) 
    # set shutter
    last_on = int(N*duty_cycle)
    assert  (last_on + transition_width) < N, "not enough samples for that "+\
            "transition width"

    data[:last_on+1, 3] = 1
    data[last_on+transition_width:, 3] = 2
    
    analog1 = np.zeros((N,), dtype=float)    
    for r in range(repeat):
        analog1[r::repeat*num_waveforms*num_phases] = 2.5
    
    # truncate data and analog1 to right length 
    data = data[offset:, :]
    analog1 = analog1[offset:]
    
    start, period = nudie.detect_table_start(analog1, repeat)
    
    shutter_info = {'last open idx': last_on - offset,
            'first closed idx' : last_on+transition_width - offset}

    tags = range(num_phases)
    tagged = nudie.tag_phases(start, period, tags, repeat,
            shutter_info=shutter_info)

    ctags = it.cycle(tags)
    creps = it.cycle(range(repeat))

    for rep in it.islice(creps, data[0, 0], repeat+data[0,0]):
        for i, tag in enumerate(it.islice(ctags, data[0, 2], num_phases + data[0,2])):
            for shutter in ['shutter open', 'shutter closed']:
                select = tagged[rep][tag][shutter]

                shutter_val = 1 if shutter == 'shutter open' else 2

                assert select.start >= 0, 'no negative index is allowed'
                assert np.all(data[select, 3] == shutter_val), \
                        'shutter was not open'
                assert np.all(data[select, 0] == rep), 'repeat was not correct'
                assert np.all(data[select, 2] == tag), 'phase was different'

                diff = np.diff(data[select, 1])
                max = np.max(np.abs(diff))
                assert np.all(np.diff(np.argwhere(diff == max)) == num_waveforms), \
                    "waveforms not appropriately periodic"

                # FIXME add test for monotonic increase of waveforms

                if shutter == 'shutter open':
                    assert data[select, 1][0] == \
                            tagged[rep][tag]['waveform shutter open']
                else:
                    assert data[select, 1][0] == \
                            tagged[rep][tag]['waveform shutter closed']

def test_several_waveforms_one_phase_offset():
    period = 6 
    num_phases = 1
    nframes = 1000
    duty_cycle = 0.5
    offset = 0
    table_start = np.array([period*i + offset for i in range(nframes//period)])
    pdb.set_trace()

    shutter_info = {'last open idx': int(nframes*duty_cycle),
            'first closed idx' : int(nframes*duty_cycle)+5}

    tags = nudie.tag_phases(table_start, period, range(num_phases), \
            shutter_info=shutter_info)

    assert tags[0][0]['waveform shutter open'] == (period - offset) % period

@pytest.mark.XFAIL
def test_one_waveform_one_phase():
    analog1 = np.zeros((1000,))

def test_exact_period():
    # Fixed bug where an assertion would fail if table_start_detect[0]
    # indicated that the first camera frame was the first phase in the 
    # waveform

    period = 4
    num_phases = 4
    nframes = 100
    duty_cycle = 0.5
    table_start = np.zeros((nframes,), dtype=int)
    table_start[::period] = np.array([4*i for i in range(nframes//period)])

    shutter_info = {'last open idx': int(nframes*duty_cycle),
            'first closed idx' : int(nframes*duty_cycle)+5}

    tags = nudie.tag_phases(table_start, period, range(num_phases), \
            shutter_info=shutter_info)

    # check that the first phase of the only waveform with the shutter open 
    # starts at index 0
    assert tags[0][0]['shutter open'].start == 0 
