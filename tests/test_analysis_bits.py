from __future__ import division
import pytest
import numpy as np
import itertools as it

import nudie

class TestDetectTableStart:
    def test_aperiodic_middle(self):
        # simulated dazzler array
        array = np.zeros((1000,))
        idx = 500
        array[idx] = 2.5

        with pytest.raises(RuntimeError):
            # should say that we found aperiodic trigger
            nudie.detect_table_start(array)

    def test_periodic_middle(self):
        period = 200
        offset = 10
        amplitude = 2.5

        array = np.zeros((1000,))
        array[offset::period] = amplitude

        found, dperiod = nudie.detect_table_start(array)

        assert dperiod == period, "incorrect period detected"
        assert np.all(np.where(array) == found), "positions are wrong"

    def test_waveform_repeat(self):
        period = 200
        offset = 10
        amplitude = 2.5
        waveform_repeat = 2
        array = np.zeros((1000,))
        for i in range(waveform_repeat):
            array[offset+i::period*waveform_repeat] = amplitude
        
        with pytest.raises(RuntimeError):
            # incorrect waveform setting
            nudie.detect_table_start(array)
        
        found, dperiod = nudie.detect_table_start(array, waveform_repeat)

        assert dperiod == period*waveform_repeat, "incorrect period found"

        correct_idx = np.zeros_like(array)
        correct_idx[offset::period*waveform_repeat] = amplitude

        assert np.all(np.where(correct_idx) == found), "positions are wrong"

    def test_lohi_truncated(self):
        # if the trigger is on on the first sample, np.diff will
        # not give a lohi transition, only a hilo. This will make
        # lohi shorter than hilo by one sample

        period = 200
        amplitude = 2.5
        offset = 0

        array = np.zeros((1010,))
        array[offset::period] = amplitude 

        found, dperiod = nudie.detect_table_start(array)

        assert dperiod == period, "incorrect period found"

        correct_idx = np.zeros_like(array)
        correct_idx[offset+period::period] = amplitude

        assert np.all(np.where(correct_idx) == found), "positions are wrong"

    def test_hilo_truncated(self):
        # similarly to above, what happens if the trigger falls on the last
        # sample?

        period = 200
        offset = 1
        amplitude = 2.5

        array = np.zeros((1002,))
        array[offset::period] = amplitude 

        found, dperiod = nudie.detect_table_start(array)

        assert dperiod == period, "incorrect period found"

        correct_idx = np.zeros_like(array)
        correct_idx[offset:-period:period] = amplitude

        assert np.all(np.where(correct_idx) == found), "positions are wrong"
    
    def test_first_and_last(self):
        # what if the trigger perfectly aligns such that the trigger is
        # both on the first and last sample?

        period = 200
        offset = 0
        amplitude = 2.5

        array = np.zeros((1001,))
        array[offset::period] = amplitude 

        found, dperiod = nudie.detect_table_start(array)

        assert dperiod == period, "incorrect period found"

        correct_idx = np.zeros_like(array)
        correct_idx[period:-period:period] = amplitude
        
        assert np.all(np.where(correct_idx) == found), "positions are wrong"

    def test_truncated_wave_repeat_begin(self):
        # dazzler trigger is on while repeating the first waveform in a
        # table. Check that detection works if only part of the
        # "waveform_repeat" triggers are detected at the beginning

        period = 100
        offset = 1
        amplitude = 2.5
        waveform_repeat = 2

        reference = np.zeros((1010,))
        for i in range(waveform_repeat):
            reference[offset+i::period*waveform_repeat] = amplitude
        
        array = reference[offset+int(waveform_repeat//2):]

        found, dperiod = nudie.detect_table_start(array,
                waveform_repeat=waveform_repeat)

        assert dperiod == period*waveform_repeat, "incorrect period found"

        correct_idx = np.zeros_like(array)
        correct_offset = period*waveform_repeat - int(waveform_repeat//2)
        correct_idx[correct_offset::period*waveform_repeat] = amplitude
        
        assert np.all(np.where(correct_idx) == found), "positions are wrong"

    def test_truncated_wave_repeat_end(self):
        # similar to above but truncated at end

        period = 100
        amplitude = 2.5
        waveform_repeat = 2

        reference = np.zeros((1002,))
        for i in range(waveform_repeat):
            reference[i::period*waveform_repeat] = amplitude 
        
        array = reference[:-int(waveform_repeat//2)]

        found, dperiod = nudie.detect_table_start(array,
                waveform_repeat=waveform_repeat)

        rep = period*waveform_repeat
        assert dperiod == rep, "incorrect period found"

        correct_idx = np.zeros_like(array)
        correct_idx[rep:-rep:rep] = amplitude 
        assert np.all(np.where(correct_idx) == found), "positions are wrong"

    def test_truncated_wave_repeat_both(self):
        # similar to above but truncated at end

        period = 100
        amplitude = 2.5
        waveform_repeat = 2

        reference = np.zeros((1002,))
        for i in range(waveform_repeat):
            reference[i::period*waveform_repeat] = amplitude 
        
        half = int(waveform_repeat//2)
        array = reference[half:-half]

        found, dperiod = nudie.detect_table_start(array,
                waveform_repeat=waveform_repeat)

        rep = period*waveform_repeat
        assert dperiod == rep, "incorrect period found"

        correct_idx = np.zeros_like(array)
        correct_idx[rep-half:-half:rep] = amplitude

        assert np.all(np.where(correct_idx) == found), "positions are wrong"
  
    def test_int_array_input(self):
        pytest.skip('implementation problem')
        # known fail, steps to reproduce
        N = 1000
        repeat = 2
        num_waveforms = 100
        analog1 = np.zeros((N,), dtype=int)    
        analog1[::repeat*num_waveforms] = 2.5
        
        start = nudie.detect_table_start(analog1, repeat)
        
        # to fix, make dtype=float

class TestSynchronizeDaqToCamera:
    def test_first_file(self):
        N = 1000
        diff = 3
        saturated = 3
        table_len = 100
        pixels = 1340
        rois = 1

        data = np.ones((rois, pixels, N), dtype='<u2')
        data[:, :, 0:saturated] = 1<<16 - 1 # simulate saturated frames
        # daq recieved fewer triggers than camera
        a1 = np.zeros((N-diff,), dtype='<u2')
        a2 = np.zeros((N-diff,), dtype='<u2')
        a1[::table_len] = 2.5
        a2[::table_len] = 2.5 # pretend that a1 and a2 are identical

        tdata, (ta1, ta2) = nudie.synchronize_daq_to_camera(data,
                analog_channels=[a1, a2], which_file='first', roi=0) #first roi

        # ROI is first, Pixels is second, Camera frames is third in data
        assert tdata.shape[0] == data.shape[1], \
                'pixel axis got trimmed!'
        assert tdata.shape[1] == data.shape[2] - diff, \
                'incorrect trimming of data'
        assert ta1.shape[0] == ta2.shape[0], \
                'analog channels have different shapes'
        assert np.allclose(ta1, ta2), \
                'synchronization between analog channels lost'
        assert tdata.shape[1] == ta1.shape[0], \
                'shape difference between analog channels and camera'

    def test_not_first(self):
        N = 1000
        diff = 3
        saturated = 3
        table_len = 100
        pixels = 1340
        rois = 1

        data = np.ones((rois, pixels, N), dtype='<u2')
        data[:, :, 0:saturated] = 1<<16 - 1 # simulate saturated frames

        # daq recieved fewer triggers than camera
        a1 = np.zeros((N+diff,), dtype='<u2')
        a2 = np.zeros((N+diff,), dtype='<u2')
        a1[::table_len] = 2.5
        a2[::table_len] = 2.5 # pretend that a1 and a2 are identical

        tdata, (ta1, ta2) = nudie.synchronize_daq_to_camera(data,
                analog_channels=[a1, a2], which_file=False, roi=0) #first roi

        # ROI is first, Pixels is second, Camera frames is third in data
        assert tdata.shape[0] == data.shape[1], \
                'pixel axis got trimmed!'
        assert tdata.shape[1] == N, \
                'incorrect trimming of data'
        assert ta1.shape[0] == ta2.shape[0], \
                'analog channels have different shapes'
        assert np.allclose(ta1, ta2), \
                'synchronization between analog channels lost'
        assert tdata.shape[1] == ta1.shape[0], \
                'shape difference between analog channels and camera'
        assert np.allclose(a1[diff:], ta1), \
                'did not truncate first `diff` points in analog channel'

    def test_input_checks(self):
        with pytest.raises(AssertionError):
            # raise wrong shape error
            nudie.synchronize_daq_to_camera(np.zeros((1, 1), dtype='<u2'))

        with pytest.raises(ValueError):
            # raise no analog channels error
            nudie.synchronize_daq_to_camera(np.zeros((1, 1, 1), dtype='<u2'))

        with pytest.raises(RuntimeError):
            # wrong number of measurements between channels
            a1, a2 = np.zeros((3,), dtype='<u2'), np.zeros((4,), dtype='<u2')
            res = nudie.synchronize_daq_to_camera(np.zeros((1, 1, 1), 
                dtype='<u2'), analog_channels=[a1,a2], which_file=False)

def test_determine_shutter_shots():
    duty_cycle = 0.9
    pixels = 1340
    rois = 1
    N = 1000
    data = np.ones((rois,pixels,N), dtype='<u2')
    data[:, :, :int(N*duty_cycle)] *= 1<<15-1
    
    with pytest.raises(AssertionError):
        # should raise because function expects data output from 
        # synchronize_daq_to_camera, which gets rid of ROI
        nudie.determine_shutter_shots(data)
    
    with pytest.raises(AssertionError):
        # shutter can't be at the beginning or the end
        nudie.determine_shutter_shots(np.squeeze(data[:, :,
            :int(N*duty_cycle)]))
    
    probe_on, probe_off, tduty_cycle = nudie.determine_shutter_shots(\
            np.squeeze(data))

    assert probe_on.stop < N*duty_cycle, \
            'found wrong shutter position'
    assert probe_off.start > N*duty_cycle, \
            'found wrong shutter position'
    assert abs(tduty_cycle - duty_cycle) < 1e-3, \
            'incorrect duty cycle estimated'

def test_load_job_issue6():
    '''this is a test for regression of issue #6 on github. currently requires
    access to the data folder, so not a portable test!'''

    stark_name = 'd1d2-linear-stark'
    stark_batch = 4
    when = '14-06-28'

    # load up pp data to use
    stark_info = next(nudie.load_job(job_name=stark_name, batch_set=[stark_batch], when=when))

    assert stark_info['batch_no'] == stark_batch, 'Incorrect assignment of batch!'
