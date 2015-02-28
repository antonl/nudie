import pytest
import numpy as np

import nudie
from matplotlib import pyplot
from scipy.signal import get_window

speed_of_light = 299.792458 # nm/fs
tolerance = 1e-8

def test_w2f_lowside():
    """check that one can pass in wavelength with blue at index 0 and red at
    the end, and that the interpolation still works.
    """
    N = 1000 # number of points
    fmin,fmax = 0.2, 0.9 # in 1000 THz
    frequency, df = np.linspace(fmin, fmax, N, retstep=True)
    wavelen = speed_of_light/frequency
    
    data = np.ones((N,)) # some dummy data to pass in
    
    tfreq, tdata, tdf =  nudie.spectrometer.wavelen_to_freq(wavelen, 
           data, ret_df=True) 

    assert abs(tdf - df) < 1e-8, "frequency step changed"
    assert np.allclose(data, tdata), "data got mangled"
    assert np.allclose(tfreq, frequency), "frequency inversion is incorrect"

def test_w2f_highside():
    """the analogous case as the one above, but frequency axis is flipped
    """
    N = 1000 # number of points
    fmin,fmax = 0.2, 0.9 # in 1000 THz
    frequency, df = np.linspace(fmin, fmax, N, retstep=True)
    wavelen = speed_of_light/frequency
    data = np.ones((N,))
    
    tfreq, tdata, tdf =  nudie.spectrometer.wavelen_to_freq(wavelen[::-1], 
           data, ret_df=True) 

    assert abs(tdf - df) < 1e-8
    assert np.allclose(data, tdata)
    assert np.allclose(tfreq, frequency)

def test_w2f_ret_df():
    """check that ret_df affects number of parameters returned
    """
    N = 1000 # number of points
    fmin,fmax = 0.2, 0.9 # in 1000 THz
    frequency, df = np.linspace(fmin, fmax, N, retstep=True)

    wavelen = speed_of_light/frequency
    data = np.ones((N,))
    
    output_true =  nudie.spectrometer.wavelen_to_freq(wavelen, data, ret_df=True) 
    assert len(output_true) == 3

    output_false =  nudie.spectrometer.wavelen_to_freq(wavelen, data,
            ret_df=False) 
    assert len(output_false) == 2

def test_f2w_ret_dwl():
    """check that ret_dwl affects number of parameters returned
    """
    N = 1000 # number of points
    fmin,fmax = 0.2, 0.9 # in 1000 THz
    frequency, df = np.linspace(fmin, fmax, N, retstep=True)
    data = np.ones((N,))
    
    output_true =  nudie.spectrometer.freq_to_wavelen(frequency, data,
            ret_dwl=True) 
    assert len(output_true) == 3

    output_false =  nudie.spectrometer.freq_to_wavelen(frequency, data,
            ret_dwl=False) 
    assert len(output_false) == 2

def test_w2f_multidimensional():
    """verify that the ax parameter works for the `wavelen_to_freq` function
    """
    N = 1000 # number of points
    fmin,fmax = 0.2, 0.9 # in 1000 THz

    frequency, df = np.linspace(fmin, fmax, N, retstep=True)
    wavelen = speed_of_light/frequency

    data = np.ones((10, 20, N, 12))
    nudie.spectrometer.wavelen_to_freq(wavelen, data, ax=2)

    with pytest.raises(ValueError):
        nudie.spectrometer.wavelen_to_freq(wavelen, data, ax=5)

    with pytest.raises(ValueError):
        nudie.spectrometer.wavelen_to_freq(wavelen[np.newaxis, :], data, ax=2)

def test_f2w_multidimensional():
    """verify that the ax parameter works for the freq_to_wavelen function
    """
    N = 1000 # number of points
    fmin,fmax = 0.2, 0.9 # in 1000 THz

    frequency, df = np.linspace(fmin, fmax, N, retstep=True)

    data = np.ones((10, 20, N, 12))
    nudie.spectrometer.freq_to_wavelen(frequency, data, ax=2)

    with pytest.raises(ValueError):
        nudie.spectrometer.freq_to_wavelen(frequency, data, ax=5)

    with pytest.raises(ValueError):
        nudie.spectrometer.freq_to_wavelen(frequency[np.newaxis, :], data, ax=2)

def test_same_speed_of_light():
    '''checks that speed of light hasn't changed'''
    assert np.allclose(nudie.spectrometer.speed_of_light, speed_of_light) 

def test_simple_wavelen():
    """check that function does not accept random `groove_density`
    """

    with pytest.raises(ValueError):
        nudie.spectrometer.simple_wavelength_axis(groove_density=120391)
    res = nudie.spectrometer.simple_wavelength_axis()
    assert res.shape[0] == 1340 # correct default number of pixels

