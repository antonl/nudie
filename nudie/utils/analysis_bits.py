'''
contains emperical voodoo that seems to make our camera and labview software
work happily together. 
'''
from __future__ import division
import logging
log = logging.getLogger('nudie.analysis_bits')
import itertools as it
import numpy as np
from pathlib import Path
from .. import SpeFile

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
                s = '`{!s}` contains invalid data. NaNs detected!'.format(file)
                log.error(s)
                raise RuntimeError(s)

            farr = arr[arr > fillval]
            return farr, farr.shape[0]
    except Exception as e:
        log.error('could not read file `{!s}`'.format(file))
        raise e

def load_analogtxt(job_name, batch_path, t2, table, loop, nchannels=2):
    '''loads the analogtxt files corresponding to a given set of t2,table,
    and loop values. Returns a list of arrays. Channel names start with 1.
    '''

    analogs = []
    for c in range(1, nchannels+1): # analog channels are 1-indexed
        s = '{t2:02d}-{table:02d}-{loop:02d}-analog{channel:d}.txt' \
                .format(t2=t2, table=table, loop=loop, channel=c)
        afile = Path(batch_path, job_name + s)
        
        log.debug('processing `{!s}`'.format(afile))

        if not afile.is_file():
            s = 'file `{!s}` does not exist'.format(afile)
            log.error(s)
            raise RuntimeError(s)
        arr, count = cleanup_analogtxt(afile) # throw away count
        analogs.append(arr)
    return analogs

def load_camera_file(job_name, batch_path, t2, table, loop, force_uint16=None):
    s = '{t2:02d}-{table:02d}-{loop:02d}.spe' \
            .format(t2=t2, table=table, loop=loop)
    camera_file = Path(batch_path, job_name + s)
    log.debug('processing `{!s}`'.format(camera_file))

    if not camera_file.is_file():
        s = 'file `{!s}` does not exist'.format(camera_file)
        log.error(s)
        raise RuntimeError(s)

    loaded = SpeFile(camera_file)
    log.debug('Got SpeFile: {!s}'.format(loaded))
    
    if force_uint16: 
        # try to fix datatype, it is setting 3
        loaded.header.datatype = 3

    # read header, make sure that ordering is correct
    # Version 2.7.6 of WinSpec doesn't seem to write these things
    # correctly...
    # FIXME: actually look at ADC gain, LowNoise/HiCap, ADCRate, etc
    if SpeFile._datatype_map[loaded.header.datatype] != np.uint16:
        log.warning('spectrum recorded with potentially incorrect datatype. '+\
                'It should probably a unsigned int16. Use force dtype ' +\
                'setting to fix this.')
    return loaded.data

def synchronize_daq_to_camera(camera_frames, analog_channels=[],
        which_file='first', roi=0):
    '''expects numpy array for camera_frames and a list of numpy arrays in 
    analog channels

    Behavior is slightly different if we're looking at the first file in a
    batch vs any other file.
    '''
    # FIXME: this function specifically needs some unit tests

    # the constant number that the DAQ lags by is set by `offby`
    # as of this commit, the number has been 3
    offby = 3

    # trim first few frames 
    trim = 3

    log.debug('synchronizing daq to camera')
    assert len(camera_frames.shape) == 3, \
            'camera frames have been squeezed()'

    if not len(analog_channels) > 0:
        s = 'didn\'t get any analog channels. Got {!r}'.format(analog_channels)
        log.error(s)
        raise ValueError(s)
    
    assert len(analog_channels[0].shape) == 1, 'analog channel should be 1D'

    if not all([x.shape == analog_channels[0].shape for x in \
        analog_channels]):
        s = 'analog channel shape differs from channel to channel. ' +\
            'This could mean there\'s a bug in the data taking ' +\
            'software. ' +\
            'Got {!s}'.format([x.shape for x in analog_channels])
        log.error(s)
        raise RuntimeError(s)

    if which_file == 'first':
        log.debug('looking at first file in a batch')
        # assume data has been loaded with SpeFile with the order being
        # [ROI index, Pixel index, Frame index]
        truncate_to = min(camera_frames.shape[2] - trim, 
                analog_channels[0].shape[0] - trim)

        log.debug('{:d}\t # of camera frames'.format(camera_frames.shape[2]))
        log.debug('{:d}\t # of analog measurements' \
                .format(analog_channels[0].shape[0]))
        log.debug('{:d}\t # offby setting'.format(offby))
        log.debug('{:d}\t # frames to trim'.format(trim))
        log.debug('{:d}\t # truncate to'.format(truncate_to))

        assert truncate_to <= analog_channels[0].shape[0] - trim
        assert truncate_to <= camera_frames.shape[2] - trim

        return camera_frames[roi, :, trim:truncate_to + trim].squeeze(), \
                [x[trim:truncate_to + trim] for x in analog_channels]
    else:
        log.debug('not the first file in a batch')

        truncate_to = min(camera_frames.shape[2] - trim, 
                analog_channels[0].shape[0] - offby - trim)

        log.debug('{:d}\t # of camera frames'.format(camera_frames.shape[2]))
        log.debug('{:d}\t # of analog measurements' \
                .format(analog_channels[0].shape[0]))
        log.debug('{:d}\t # offby setting'.format(offby))
        log.debug('{:d}\t # frames to trim'.format(trim))
        log.debug('{:d}\t # truncate to'.format(truncate_to))

        assert truncate_to <= analog_channels[0].shape[0] - offby - trim
        assert truncate_to <= camera_frames.shape[2] - trim

        # throw away the first offby measurements on the daq
        # note that the order is different for throwing away measurements
        # in comparison to the previous case
        return camera_frames[roi, :, trim:truncate_to + trim].squeeze(), \
                [x[trim+offby:truncate_to+offby+trim] for x in analog_channels]

def determine_shutter_shots(camera_data):
    '''tries to determine a range in the camera frames that have the shutter on
    or off'''

    # throw away `transition_width` shots around the found position
    transition_width = 5    

    assert len(camera_data.shape) == 2, 'got camera data with ROI'
    power_trace = camera_data.mean(axis=0)

    # potential FIXME: this will find the maximal change. What if shutter is opened
    # multiple times? 
    shutter_start = np.argmax(abs(np.diff(power_trace)))
    
    assert all([shutter_start != 0, shutter_start != len(power_trace)-1]), \
            'shutter start found at the beginning or the end of the trace'+\
            '. This is probably horribly wrong.'

    assert camera_data.shape[1] > 2*transition_width, 'camera trace too short!' 

    duty_cycle = shutter_start/(power_trace.shape[0]-1)

    s = 'found shutter at {:d}/{:d}. This corresponds to a shutter duty ' +\
            'cycle of {:.0%}. Does that look right?'
    log.info(s.format(shutter_start, len(power_trace)-1, duty_cycle))

    probe_on = slice(None, shutter_start - transition_width)
    probe_off = slice(shutter_start + transition_width, None)
    return probe_on, probe_off, duty_cycle

def detect_table_start(array, waveform_repeat=1):
    '''detects peaks output when the Dazzler table starts at the begining by
    looking at the discrete difference of the array.
    '''

    diff = 2 # at least 2 Volt difference
    darr = np.diff(array)
    lohi, hilo = np.argwhere(darr > 2), np.argwhere(darr < -2)

    assert len(lohi) == len(hilo), 'spike showed up as the first or last signal'
    if not np.allclose(hilo-lohi, waveform_repeat):
        s = 'detected incorrect repeat or a break in periodicity ' +\
            'of the table. Check that the DAQ card cable is connected ' + \
            'and dazzler synchronization is working correctly. ' + \
            'Alternatively, check the waveform repeat setting. ' + \
            'Waveform repeat is set to {:d}.'.format(waveform_repeat)
        log.error(s)
        raise RuntimeError(s)
    # diff returns forward difference. Add one for actual peak position 
    return np.squeeze(lohi)+1 

def tag_phases(table_start_detect, waveform_range, waveform_repeat=1,
        trim_range=None):
    '''tag camera frames based on the number of waveforms and waveform repeat'''
    pass

