from __future__ import division, absolute_import

import logging
log = logging.getLogger('nudie.spectrometer')
import numpy as np
from scipy.interpolate import interp1d

speed_of_light = 299.792458 # nm/fs

hg_ar_lines = np.array( [253.652, 296.728, 302.150, 312.567, 313.155, 334.148, 
    365.015, 404.656, 407.783, 435.833, 546.074, 576.960, 579.066, 696.543, 
    706.722, 710.748, 727.294, 738.393, 750.387, 763.511, 772.376, 794.818, 
    800.616, 811.531, 826.452, 842.465, 852.144, 866.794, 912.297, 922.450])

def simple_wavelength_axis(groove_density=600, center_wavelength=650, 
        num_pixels=1340):
    '''uses Horiba's listing of typical spectral bandwidth to convert a groove
    density setting to a bandwidth'''

    # these values are groove density vs spectral bandwidth in nm, taken from 
    # manual
    bandwidth_map = { \
            150: 496,
            300: 248,
            600: 124,
            900: 83,
            1200: 62,
            1800: 41,
            2400: 31
            }

    if groove_density not in bandwidth_map.keys():
        s = 'could not find a mapping from that groove density to a ' +\
                'spectral bandwidth. Valid options are {!s}' \
                .format(bandwidth_map)
        log.error(s)
        raise ValueError(s)

    bw = bandwidth_map[groove_density]
    return np.linspace(center_wavelength - bw/2, center_wavelength + bw/2, 
            num_pixels)

def wavelen_to_freq(axis, data, ret_df=False):
    '''convert wavelength in nm to frequency in Hz'''

    log.debug('converting wavelength to frequency')
    
    assert len(axis.shape) == 1, 'wavelength axis should be 1D'
    assert len(data.shape) == 1, 'data assumed to be 1D'

    if axis[-1] > axis[0]: # make wavelength go from red to blue
        log.debug('flipping axis and data')
        axis = axis[::-1] 
        data = data[::-1] 
    
    dat_freq = speed_of_light/axis
    
    # former bug: data needs to be inverted when in frequency space
    interpolator = interp1d(dat_freq, data, kind='cubic')

    freq, df = np.linspace(dat_freq[0], dat_freq[-1], data.shape[0], retstep=True)

    if ret_df:
        return freq, interpolator(freq), df
    else:
        return freq, interpolator(freq)

def freq_to_wavelen(axis, data, ret_dwl=False):
    '''convert wavelength in nm to frequency in Hz'''

    log.debug('converting frequency to wavelength')
    
    assert len(axis.shape) == 1, 'wavelength axis should be 1D'
    assert len(data.shape) == 1, 'data assumed to be 1D'

    if axis[-1] > axis[0]: # make frequency go from blue to red
        log.debug('flipping axis and data')
        axis = axis[::-1] 
        data = data[::-1] 
    
    dat_wl = speed_of_light/axis
    
    # former bug: data needs to be inverted when in frequency space
    interpolator = interp1d(dat_wl, data, kind='cubic')

    wl, dwl = np.linspace(dat_wl[0], dat_wl[-1], data.shape[0], retstep=True)

    if ret_dwl:
        return wl, interpolator(wl), dwl 
    else:
        return wl, interpolator(wl)
