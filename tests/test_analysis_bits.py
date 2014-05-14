from __future__ import division
import pytest
import numpy as np

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
