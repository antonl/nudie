import pytest
import numpy as np

import nudie
speed_of_light = 299.792458 # nm/fs

def test_wavelen_to_freq_lowside():
    '''converts wavelength in nm to frequency in Hz, 
    wavelength from blue to red
    '''
    
    frequency, df = np.linspace(1, 100, 100, retstep=True)
    wavelen = speed_of_light/frequency
    
    data = np.ones((100,))
    
    tfreq, tdata, tdf =  nudie.spectrometer.wavelen_to_freq(wavelen, 
           data, ret_df=True) 

    assert abs(tdf - df) < 1e-8
    assert np.allclose(data, tdata)
    assert np.allclose(tfreq, frequency)

def test_wavelen_to_freq_highside():
    '''converts wavelength in nm to frequency in Hz, 
    wavelength from red to blue
    '''
    frequency, df = np.linspace(1, 100, 100, retstep=True)
    wavelen = speed_of_light/frequency
    
    data = np.ones((100,))
    
    tfreq, tdata, tdf =  nudie.spectrometer.wavelen_to_freq(wavelen[::-1], 
           data, ret_df=True) 

    assert abs(tdf - df) < 1e-8
    assert np.allclose(data, tdata)
    assert np.allclose(tfreq, frequency)

def test_same_speed_of_light():
    '''checks that speed of light hasn't changed'''
    speed_of_light = 10.
    frequency, df = np.linspace(1, 100, 100, retstep=True)
    wavelen = speed_of_light/frequency
    
    data = np.ones((100,))
    
    tfreq, tdata, tdf =  nudie.spectrometer.wavelen_to_freq(wavelen[::-1], 
           data, ret_df=True) 

    assert not (abs(tdf - df) < 1e-8)
    assert np.allclose(data, tdata)
    assert not np.allclose(tfreq, frequency)

def test_simple_wavelen():
    '''just check that one value is correct'''

    with pytest.raises(ValueError):
        nudie.spectrometer.simple_wavelength_axis(groove_density=120391)
    
    res = nudie.spectrometer.simple_wavelength_axis()
    assert res.shape[0] == 1340 # correct default number of pixels
    
