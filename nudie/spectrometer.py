from __future__ import division, absolute_import

import logging
log = logging.getLogger('nudie.spectrometer')
import numpy as np
from scipy.interpolate import interp1d

speed_of_light = 299.792458 # nm/fs
"""the speed of light in nanometers per femtosecond

References
----------
.. [1] http://physics.nist.gov/cgi-bin/cuu/Value?c
"""

hg_lines = np.array([184.91, 194.17, 226.22, 237.83, 248.2, 253.65, 265.2,
    275.28, 280.35, 289.36, 296.73, 302.15, 312.57, 313.17, 334.15, 365.02,
    365.48, 366.33, 404.66, 407.78, 433.92, 434.75, 435.84, 491.6, 546.07, 
    576.96, 579.07, ])
ne_lines = np.array([336.99, 341.79, 344.77, 345.42, 346.66, 347.26, 349.81,
    350.12, 351.52, 352.05, 359.35, 360.02, 363.37, 470.44, 503.78, 508.04,
    511.65, 514.5, 520.39, 533.08, 534.11, 534.33, 540.06, 565.67, 574.83,
    576.44, 580.45, 582.02, 585.25, 588.19, 590.25, 594.48, 597.55, 598.79,
    603.0, 607.43, 609.62, 612.85, 614.31, 616.36, 618.22, 621.73, 626.65,
    630.48, 633.44, 638.3, 640.23, 650.65, 653.29, 659.9, 665.21, 667.83,
    671.7, 692.95, 702.41, 703.24, 705.3, 705.91, 717.39, 724.52, 743.89,
    747.24, 748.89, 753.58, 754.41, 794.32, 808.25, 813.64, 830.03, 837.76,
    841.84, 849.54])
ar_lines = np.array([355.43, 394.9, 404.44, 415.86, 416.4, 418.19, 419.1,
    419.8, 420.07, 425.12, 425.94, 426.63, 427.22, 427.4, 432.0, 433.36,
    434.52, 641.63, 667.73, 675.28, 687.13, 693.77, 696.54, 703.03, 706.72,
    714.70, 727.29, 737.21, 738.4, 750.39, 751.46, 763.51, 772.38, 794.82,
    800.62, 801.48, 810.37, 811.53, 826.45, 840.82, 842.46, ])

hg_ar_lines = np.sort(np.concatenate([hg_lines, ar_lines]))

'''
hg_ar_lines = np.array( [
    184.91, 194.17, 226.22, 237.83, 248.2, 253.652, 296.728, 302.150, 312.567, 
    313.155, 334.148, 365.015, 404.656, 407.783, 435.833, 546.074, 576.960, 
    579.066, 696.543, 706.722, 710.748, 727.294, 738.393, 750.387, 763.511, 
    772.376, 794.818, 800.616, 811.531, 826.452, 842.465, 852.144, 866.794, 
    912.297, 922.450])
'''
"""listing of the wavelengths of the strongest Mercury Argon lines in
nanometers

This is a list of the strongest Mercury-Argon lines from pencil light 
sources. The list was obtained from NIST [1]_[2]_ and the Ocean Optics
website [3]_.

References
----------
.. [1] http://physics.nist.gov/PhysRefData/Handbook/Tables/argontable2.htm
.. [2] http://physics.nist.gov/PhysRefData/Handbook/Tables/mercurytable2.htm
.. [3] http://oceanoptics.com/product/hg-1/
"""
def simple_wavelength_axis(groove_density=600, center_wavelength=650, 
        num_pixels=1340):
    """generate a approximate wavelength axis given bandwidth and central
    wavelength

    This function allows you to quickly generate a wavelength axis without
    taking a calibration spectrum with the Mercury-Argon source. This
    wavelength axis will be evenly-spaced in wavelength and will be have the
    bandwidth given by the Horiba spectrometer manual. This should *not* be
    used for real analysis; it is just for quick-and-dirty calculations.

    Parameters
    ----------
    groove_density : int, optional
        
        An integer representing the groove density in lines per mm. Shows up in
        the spectrometer control software as, e.g., 600g. Defaults to 600.
    
    center_wavelength : float, optional

        The central wavelength of the spectrometer in nanometers. Defaults to
        650.

    num_pixels : int, optional

        The number of pixels of the camera. This parameter controls the number
        of points that will be generated in the array. 

    Returns
    -------
    wavelengths : float array
        
        The mapping of pixel to wavelength. 

    Raises
    ------
    ValueError 
        
        If the groove density is not one of the ones listed in the manual.
    """

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
    wavelengths = np.linspace(center_wavelength - bw/2, 
            center_wavelength + bw/2, num_pixels)
    return wavelengths

def wavelen_to_freq(axis, data, ret_df=False, ax=-1):
    """linearly interpolates data from equal-wavelength sampling to
    equal-frequency sampling
    
    This function is used to convert raw data from the spectrometer to
    equal-frequency sampling. This is used as a first step before
    Fourier-transformation so that the FFT'ed data is linear in time-delay. The
    frequency units are in 1000 THz.

    This interpolation is done to the same number of points. That is, if there
    were 1340 equally-spaced wavelengths, there will be 1340 equally-spaced
    frequencies.

    Parameters
    ----------
    axis : float array, (N,)
        
        Represents the wavelength axis of length N. Must be 1D.

    data : float array, (..., N)
        
        Data hypercube to linearly interpolate along some axis. At least one of
        the axes must be of length N. That's the one you want to interpolate
        over.
    
    ret_df : boolean, optional

        Determines whether or not to return the frequency step as an extra
        parameter. Defaults to false.

    ax : int, optional

        The index of the axis over which to interpolate. This axis must be the
        same length as the axis array. Defaults to -1, the last axis.

    Returns
    -------
    frequency : float array, (N,)
        
        The interpolated frequency axis of length N in units of 1000 THz.

    interpolated_data : float array, (..., N)

        Interpolated data array.

    df : float, optional

        The frequency step of the :param:`frequency` array. Returned when
        :param:`ret_df` is True

    See Also
    --------
    freq_to_wavelen : frequency to wavelength data interpolation
    """

    log.debug('converting wavelength to frequency')
    
    if len(axis.shape) != 1:
        s = 'wavelength axis should be 1D'
        log.error(s)
        raise ValueError(s)
    
    if ax > len(data.shape) - 1: 
        s = 'ax index longer than data array dimension'
        log.error(s)
        raise ValueError(s)
    
    if axis[-1] > axis[0]: # make wavelength go from red to blue
        log.debug('flipping axis and data')

        if ax is -1: # avoid -1, doesn't work with flipping algorithm
            ax = len(data.shape)-1
        
        # generalize to arbitrary axis flip
        ind = tuple([slice(None, None, -1) if ax == i else slice(None) \
                for i in range(len(data.shape))])

        axis = axis[::-1] 
        data = data[ind]        
    
    dat_freq = speed_of_light/axis
    
    # former bug: data needs to be inverted when in frequency space
    interpolator = interp1d(dat_freq, data, kind='linear', axis=ax)

    # former bug: issue #7
    freq, df = np.linspace(dat_freq[0], dat_freq[-1], axis.shape[-1], retstep=True)

    if ret_df:
        return freq, interpolator(freq), df
    else:
        return freq, interpolator(freq)

def freq_to_wavelen(axis, data, ret_dwl=False, ax=-1):
    """linearly interpolates data from equal-wavelength sampling to
    equal-frequency sampling

    The inverse operation of wavelen_to_freq. Returns wavelengths in 
    units of nm.

    Parameters
    ----------
    axis : float array, (N,)
        
        Represents the frequency axis of length N. Must be 1D.

    data : float array, (..., N)
        
        Data hypercube to linearly interpolate along some axis. At least one of
        the axes must be of length N. That's the one you want to interpolate
        over.
    
    ret_dwl : boolean, optional

        Determines whether or not to return the wavelength step as an extra
        parameter. Defaults to false.

    ax : int, optional

        The index of the axis over which to interpolate. This axis must be the
        same length as the axis array. Defaults to -1, the last axis.

    Returns
    -------
    wavelength : float array, (N,)
        
        The interpolated wavelength axis of length N in units of nm.

    interpolated_data : float array, (..., N)

        Interpolated data array.

    dwl : float, optional
        The wavelength step of the `wavelength`. Returned when
        `ret_df` is False

    See Also
    --------
    wavelen_to_freq : wavelength to frequency data interpolation
    """
    log.debug('converting frequency to wavelength')
    
    if len(axis.shape) != 1:
        s = 'wavelength axis should be 1D'
        log.error(s)
        raise ValueError(s)

    if ax > len(data.shape) - 1: 
        s = 'ax index longer than data array dimension'
        log.error(s)
        raise ValueError(s)

    if axis[-1] > axis[0]: # make frequency go from blue to red
        log.debug('flipping axis and data')

        if ax is -1: # avoid -1, doesn't work with flipping algorithm
            ax = len(data.shape)-1

        # generalize to arbitrary axis flip
        ind = tuple([slice(None, None, -1) if ax == i else slice(None) \
                for i in range(len(data.shape))])
        axis = axis[::-1] 
        data = data[ind]

    dat_wl = speed_of_light/axis
    
    # former bug: data needs to be inverted when in frequency space
    interpolator = interp1d(dat_wl, data, kind='linear', axis=ax)

    # former bug: issue #7
    wl, dwl = np.linspace(dat_wl[0], dat_wl[-1], axis.shape[-1], retstep=True)

    if ret_dwl:
        return wl, interpolator(wl), dwl 
    else:
        return wl, interpolator(wl)
