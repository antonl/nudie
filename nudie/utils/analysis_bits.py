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
from scipy.signal import get_window, find_peaks_cwt
from scipy.fftpack import fft, fftshift, fftfreq
from .. import wavelen_to_freq

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

    assert all([x.shape[0] > offby for x in analog_channels]), \
            'offby is longer than recorded analog channels'

    if which_file == 'first':
        log.debug('looking at first file in a batch')
        # assume data has been loaded with SpeFile with the order being
        # [ROI index, Pixel index, Frame index]
        truncate_to = min(camera_frames.shape[2],
                analog_channels[0].shape[0])

        log.debug('{:d}\t # of camera frames'.format(camera_frames.shape[2]))
        log.debug('{:d}\t # of analog measurements' \
                .format(analog_channels[0].shape[0]))
        log.debug('{:d}\t # offby setting'.format(offby))
        log.debug('{:d}\t # truncate to'.format(truncate_to))

        assert truncate_to <= analog_channels[0].shape[0]
        assert truncate_to <= camera_frames.shape[2]

        return camera_frames[roi, :, :truncate_to].squeeze(), \
                [x[:truncate_to] for x in analog_channels]
    else:
        log.debug('not the first file in a batch')

        truncate_to = min(camera_frames.shape[2],
                analog_channels[0].shape[0] - offby)

        log.debug('{:d}\t # of camera frames'.format(camera_frames.shape[2]))
        log.debug('{:d}\t # of analog measurements' \
                .format(analog_channels[0].shape[0]))
        log.debug('{:d}\t # offby setting'.format(offby))
        log.debug('{:d}\t # truncate to'.format(truncate_to))

        assert truncate_to <= analog_channels[0].shape[0] - offby
        assert truncate_to <= camera_frames.shape[2]

        # throw away the first offby measurements on the daq
        # note that the order is different for throwing away measurements
        # in comparison to the previous case
        return camera_frames[roi, :, :truncate_to].squeeze(), \
                [x[offby:truncate_to+offby] for x in analog_channels]

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

    diff = 1.9 # ~2 Volt difference should be the TTL trigger
    darr = np.diff(array)
    lohi, hilo = np.argwhere(darr > 2), np.argwhere(darr < -2)

    # should be short cut evaluation, so second statement assumes same length
    # arrays
    if len(lohi) != len(hilo) or not np.all(lohi < hilo):
        log.warning('spike showed up as the first or last signal')
        # check which one is longer, and which comes first 
        mlen = min(len(lohi), len(hilo))
        log.warning('minimal length is {:d}'.format(mlen))
        if len(lohi) > mlen:
            # order is correct, but hilo is missing a point
            # truncate lohi at the end
            log.warning('truncated lohi to match hilo')
            lohi = lohi[:-1]
            assert len(lohi) == len(hilo), 'did not help'
        elif len(hilo) > mlen:
            # order is incorrect, must be missing first point
            log.warning('truncated hilo to match lohi')
            hilo = hilo[1:]
            assert len(lohi) == len(hilo), 'did not help'
        elif not np.all(lohi[:mlen] < hilo[:mlen]):
            log.warning('both sides missing, shift over one')
            lohi, hilo = lohi[:-1], hilo[1:]
            assert len(lohi) == len(hilo), 'did not help'
        else:
            assert False, "should not happen!"

        mlen = max(len(lohi), len(hilo))
        log.warning('truncating to {:d}'.format(mlen))
        lohi, hilo = lohi[:mlen], hilo[:mlen]

    if lohi.shape[0] == 1:
        s = 'only one complete dazzler table found. Are you sure' +\
                ' you integrated for long enough?'
        log.error(s)
        raise RuntimeError(s)
    else:
        period = int(lohi[1] - lohi[0])

    if not np.allclose(hilo-lohi, waveform_repeat) or \
        not np.allclose(np.diff(lohi),  period):

        s = 'detected incorrect repeat or a break in periodicity ' +\
            'of the table. Check that the DAQ card cable is connected ' + \
            'and dazzler synchronization is working correctly. ' + \
            'Alternatively, check the waveform repeat setting. ' + \
            'Waveform repeat is set to {:d}.'.format(waveform_repeat)

        log.error(s)
        raise RuntimeError(s)
        
    # diff returns forward difference. Add one for actual peak position 
    return np.squeeze(lohi)+1, period

def trim_all(cdata, ais, trim_to=slice(10, -10)):
    '''trim all outputs to given slice'''
    assert (trim_to.step == None) or (trim_to.step == 1), +\
            'trim_to shouldn\'t be strided.'

    assert len(cdata.shape) == 2, 'camera shouldn\'t have ROI anymore'
    tai = []
    for ai in ais: tai.append(ai[trim_to])
    return cdata[:, trim_to], tai

def tag_phases(table_start_detect, period, tags, waveform_repeat=1,
        shutter_info=None):
    '''tag camera frames based on the number of waveforms and waveform repeat'''

    nphases = len(tags)
    if nphases < 1:
        s = 'need at least one waveform tag. Supply a list with the ' +\
                'waveforms in the order that they appear in the dazzler ' +\
                'table.' 
        log.error(s)
        raise ValueError(s)
    
    if not (period % waveform_repeat == 0):
        s = 'waveform repeat does not divide period without remainder! ' +\
                'Are you sure waveform_repeat and period are correct? ' +\
                'Waveform_repeat: {:d}\tPeriod: {:d}'.format(waveform_repeat,
                        period)
        log.error(s)
        raise ValueError(s)

    nwaveforms = period // (nphases*waveform_repeat)

    # FIXME: this isn't quite right. I should probably group the shutter open
    # and shutter closed shots in the same pass. I can then subtract the pump
    # scatter that is specific to each phase.
    if shutter_info is None:
        log.debug('shutter shots not taken into account when tagging ' +\
                'phases')
        raise NotImplementedError('please pass shutter info')
    elif shutter_info and \
            not all([x in shutter_info.keys() for x in ['last open idx', 
                'first closed idx']]):
        s = 'invalid shutter info dict. Need \'last open idx\' and ' +\
                '\'first closed idx\' keys'
        log.error(s)
        raise ValueError(s)
    else:
        log.debug('using shutter info, it has the right keys. {!s}'\
                .format(shutter_info))
    
    period_index = period - table_start_detect[0]

    assert period_index >= 0, \
        'had partial table at the beginning that is longer ' +\
        'than total periodicity'

    if not (period % nphases == 0):
        s = 'number of tags does not divide period without remainder! ' +\
                'Are you sure that you set the right number of tags? ' +\
                'Tags: {!s}\tNum Tags: {:d}\tPeriod: {:d}'.format(tags,
                        nphases, period)
        log.error(s)
        raise ValueError(s)

    # Tee is very important, Otherwise we move the cycler forward 
    # for each rep and lose synchronization
    ctags = it.tee(it.cycle(tags), waveform_repeat)
    crep = it.cycle(range(waveform_repeat))
    # What is the repeat setting?  
    skip_repeats = int((period_index + 1) % waveform_repeat)
    # What is the current phase?
    skip_phases = int((period_index + 1 - skip_repeats // (waveform_repeat)))
    
    if skip_repeats != 0: 
        next(ctags[0]) # roll over phase counter if reps roll over

    # this is the earliest waveform that the camera frame could be
    min_waveform = (period_index // nwaveforms) % nwaveforms

    tagged = {}
    for rep in it.islice(crep, skip_repeats, waveform_repeat + skip_repeats):
        tmp = {}

        for i, tag in enumerate(it.islice(ctags[rep], skip_phases, nphases+skip_phases)):
            repeats = (skip_repeats - rep) % waveform_repeat
            offset = repeats + i*waveform_repeat
            open = slice(offset, shutter_info['last open idx'],
                    waveform_repeat*nphases)
            k = (shutter_info['first closed idx'] - offset) // nwaveforms 
            closed = slice(offset + (k+1)*waveform_repeat*nphases, 
                    None, waveform_repeat*nphases)
            first_waveform = min_waveform + (skip_repeats + \
                    waveform_repeat*skip_phases + offset) // nwaveforms            
            tmp[tag] = {'shutter open': open,
                    'shutter closed': closed,
                    'waveform shutter open': first_waveform,
                    'waveform shutter closed': (first_waveform +\
                        k % nwaveforms),
                    }
        tagged.update({rep: tmp})
    tagged.update({'nwaveforms': nwaveforms, 'min_waveform': min_waveform})
    return tagged

def identify_prd_peak(wl, data, window=None, axes=None):
    assert len(data.shape) == 1, 'expected spectrum averaged over ' +\
            'camera frames'
    peak_widths = np.arange(5, 200, 5)
    threshold = 0.001 # fraction of DC peak 

    if window:
        # for future
        raise NotImplementedError('don\'t know how to use a custom window')
    
    window = get_window(('kaiser', 11), len(prd), fftbins=False)

    freq, prd, df = wavelen_to_freq(wl, data, ret_df=True)
    time = fftshift(fftfreq(len(freq), df))
    ft = fftshift(abs(fft(prd*window)))

    res = np.array(find_peaks_cwt(ft, peak_widths, min_length=3))
    select = ft[res] > threshold*np.max(ft)
    prd_idx = res[select][-1]

    if axes:
        # going to plot stuff to axes
        axes.plot(time, ft)
        next_highest = ft[res[select][-1]]
        axes.set_ylim(0, 1.1*next_highest)
        #axes.set_xlim(time[prd_idx]-, time[prd_idx])
        axes.vlines(time[res], 0, next_highest, color='r')

    log.info('found PRD at {:.1f} fs'.format(time[prd_idx]))

    return prd_idx, time

def make_6phase_cycler(phase_pairs):
    '''given pairs of phases for pump 1 and pump 2, generates the 
    phase inverting matrix'''

    N = len(phase_pairs)
    A = np.zeros((N,N), dtype=complex)
    for i,(x, y) in enumerate(phase_pairs):
        A[i, 0] = np.exp(1j*(x-y))
        A[i, 1] = np.exp(-1j*(x-y))
        A[i, 2] = 1
    mat_inv = np.inv(mat)

    def wrapped(*SI):
        # TODO: test this. not sure that the tensor dot works like I expect
        assert len(SI) == 6, 'need 6 phases to do 6 phase cycling'
        R1, NR1, TG1, R2, NR2, TG2 = SI
        R = 0.5*(R1 + R2)
        NR = 0.5*(NR1 + NR2)
        TG = 0.5*(TG1 + TG2)
        res = np.tensordot(mat_inv, np.array([R, NR, TG]), [[1],[0]])
        return res[0, :], res[1,:], res[2,:]
    return wrapped
