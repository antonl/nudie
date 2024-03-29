#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 22:14:42 2016

@author: aloukian
"""

import xarray as xr
import traitlets
import pathlib
import toolz
import itertools as it
import re
import numpy as np
import pandas as pd
from multiprocessing import Pool
from winspec import SpeFile

starmap = toolz.curry(it.starmap)
rcompose = lambda *fns: toolz.comp(*list(reversed(fns)))
curry = toolz.curry
map = toolz.curry(map)

'''
Test implementation of work graph that uses signals to represent input and
output channels.
'''

PhaseCycles = {
    '2D': [(0., 0.), (1., 1.), (0, 0.6), (1., 1.6), (0, 1.3), (1., 2.3)],
    '2DESS': [(0., 0.), (1., 1.), (0, 0.6), (1., 1.6), (0, 1.3), (1., 2.3)],
    'Linear Stark': ['none1', 'zero', 'none2', 'pipi'],
    'Pump Probe': ['none1', 'zero', 'none2', 'pipi'],
    'Transient Grating': ['none1', 'zero', 'none2', 'pipi'],
    'TGESS': ['none1', 'zero', 'none2', 'pipi']
}

def cleanup_analogtxt(file):
    '''remove fill values in the analog txt files'''
    fillval = -1
    try:
        file = pathlib.Path(file)
        with file.open(mode='rb') as f:
            arr = np.genfromtxt(f, dtype=float)

            assert len(arr.shape) == 1, "should be single column file"
            if not np.all(np.not_equal(arr, np.nan)):
                s = '`{!s}` contains invalid data. NaNs detected!'.format(file)
                raise RuntimeError(s)

            farr = arr[arr > fillval]
            return farr, farr.shape[0]
    except FileNotFoundError as e:
        raise e

def load_camera_file(spe_file, force_uint16=None):
    camera_file = pathlib.Path(spe_file)
   
    if not camera_file.is_file():
        s = 'file `{!s}` does not exist'.format(camera_file)
        raise RuntimeError(s)

    loaded = SpeFile(str(camera_file))  
    if force_uint16:
        # try to fix datatype, it is setting 3
        loaded.header.datatype = 3
        
    return loaded.data

def synchronize_daq_to_camera(camera_frames, analog_channels,
        which_file='first', roi=0, offby=3):
    '''expects numpy array for camera_frames and a list of numpy arrays in 
    analog channels

    Behavior is slightly different if we're looking at the first file in a
    batch vs any other file.
    '''

    assert len(camera_frames.shape) == 3, \
            'camera frames have been squeezed()'
    
    assert len(analog_channels[0].shape) == 1, 'analog channel should be 1D'

    if not all([x.shape == analog_channels[0].shape for x in \
        analog_channels]):
        s = 'analog channel shape differs from channel to channel. ' +\
            'This could mean there\'s a bug in the data taking ' +\
            'software. ' +\
            'Got {!s}'.format([x.shape for x in analog_channels])
        raise RuntimeError(s)

    assert all([x.shape[0] > offby for x in analog_channels]), \
            'offby is longer than recorded analog channels'

    if which_file == 'first':
        # assume data has been loaded with SpeFile with the order being
        # [Frame index, pixel index, roi index]
        truncate_to = min(camera_frames.shape[0],
                analog_channels[0].shape[0])

        assert truncate_to <= analog_channels[0].shape[0]
        assert truncate_to <= camera_frames.shape[0]

        return camera_frames[:truncate_to, :, roi].squeeze(), \
                [x[:truncate_to] for x in analog_channels]
    else:
        truncate_to = min(camera_frames.shape[0],
                analog_channels[0].shape[0] - offby)

        assert truncate_to <= analog_channels[0].shape[0] - offby
        assert truncate_to <= camera_frames.shape[0]

        # throw away the first offby measurements on the daq
        # note that the order is different for throwing away measurements
        # in comparison to the previous case
        return camera_frames[:truncate_to, :, roi].squeeze(), \
                [x[offby:truncate_to+offby] for x in analog_channels]

class Worker(traitlets.HasTraits):
    '''a functor object representing some unit of work'''
    name = 'Worker'

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __str__(self):
        return '{}'.format(self.name)

    def __call__(self, *args, **kwargs):
        return None


class BatchLoader(Worker):
    '''initializes loading of a job given by a path
    '''
    batch_path = traitlets.Unicode(help='location of the batch folder on disk', 
        allow_none=False)
    
    def __init__(self, batch_path):
        self.batch_path = batch_path
    
    def __call__(self):
        loaded_path = pathlib.Path(self.batch_path)
        
        paths = list(loaded_path.glob('*.spe'))
        
        folder_pattern = re.compile('(?P<job_name>[\w-]+)-batch(?P<batch_no>\d+)')
        
        m = folder_pattern.match(loaded_path.name)
        
        if m is None:
            raise ValueError('This is probably not a data folder.')
            
        job_name = m.group('job_name')
        batch_no = m.group('batch_no')
        
        file_pattern = re.compile(job_name + r'(\d+)-(\d+)-(\d+)\.spe')
        
        max_t2, max_table, max_loop = 0, 0, 0
        
        for p in paths:
            m = file_pattern.match(p.name)
            #print(m.group(1), m.group(2), m.group(3))
            nt2, ntables, nloop = int(m.group(1)), int(m.group(2)), int(m.group(3))
            
            max_t2 = max([max_t2, nt2])
            max_table = max([max_table, ntables])
            max_loop = max([max_loop, nloop])
        
        s = 'Loading job <b>{job_name}-batch{batch_no}</b>.<br /><br />' +\
            'Found {n} SPE files, with {nt2} t2 points, '+\
            'in {ntables} tables and with {nloop} loops.'
        s = s.format(job_name=job_name, 
                                   batch_no=batch_no,
                                   n=len(paths),
                                   nt2=max_t2+1, 
                                   ntables=max_table+1, 
                                   nloop=max_loop+1)
        
        t1vals = np.genfromtxt(str(loaded_path / 't1pos.txt'), dtype=float)
        t2vals = np.genfromtxt(str(loaded_path / 't2pos.txt'), dtype=float).reshape((-1,2))
        
        for indexes in it.product(range(max_t2+1), 
                                  range(max_table+1),
                                  range(max_loop+1)):
            
            fn = '{:s}{:02d}-{:02d}-{:02d}'.format(job_name, 
                indexes[0], indexes[1], indexes[2])
            spe_file = str(loaded_path / (fn + '.spe'))
            analog1 = str(loaded_path / (fn + '-analog1.txt'))
            analog2 = str(loaded_path / (fn + '-analog2.txt'))
            yield spe_file, [analog1, analog2], indexes, t1vals, t2vals, \
                loaded_path.name, (max_t2+1, max_table+1, max_loop+1)
    
class CallTester(Worker):
    def __call__(self, args):
        print(args)
        yield
        
class DataChunkLoader(Worker):
    '''loads a single SPE file and its associated analog channels
    '''

    trim_to = traitlets.Instance(klass=slice, args=(10, None), 
        help='range of indexes to throw away when loading a camera file' +\
            '; used for removing saturated frames at the beginning of ' +\
            'acquisition').tag(config=True)

    offby = traitlets.Int(default_value=3,
        help='number of laser shots offset between camera and DAQ after the '\
        + 'first file.')

    #inputs = namedtuple('inputs', ['spe_file', 'analogs', 'indexes'])

    def __call__(self, spe_file, analogs, indexes, t1vals, t2vals, dset_name,
            max_indexes):
        '''perform data conversion given the proper files
        '''

        assert all([idx > -1 for idx in indexes]), "indexes must be non-negative"

        t2_idx = indexes[0]
        table_idx = indexes[1]
        loop_idx = indexes[2]

        analog_channels = [cleanup_analogtxt(filename)[0] for filename in analogs]
        spe_data = load_camera_file(spe_file, force_uint16=True)
        
        first = 'first' if all([idx == 0 for idx in indexes]) else 'false'
        camera_data, analogs = synchronize_daq_to_camera(spe_data, analog_channels,
                                                         which_file=first, offby=self.offby)
        
        coords = {'laser_shot': np.arange(camera_data.shape[0]),
                  'pixel': np.arange(camera_data.shape[1])
                  }

        data_vars = {'spectrometer_data': (['laser_shot', 'pixel'], camera_data),
                     'achan1': (['laser_shot'], analogs[0]),
                     'achan2': (['laser_shot'], analogs[1])}

        dset = xr.Dataset(data_vars, coords=coords)
        
        dset.attrs['t2idx'] = t2_idx
        dset.attrs['tableidx'] = table_idx
        dset.attrs['loopidx'] = loop_idx
        dset.attrs['t2vals'] = -t2vals
        dset.attrs['t1vals'] = -t1vals
        dset.attrs['name'] = dset_name
        dset.attrs['maxidx'] = max_indexes
        
        return dset

class PhaseCycler(Worker):
    phase_cycles = traitlets.List(default_value=['frame'], minlen=1,
        help='iterable containing phase-cycle labels').tag(config=True)

    nwaveforms = traitlets.Int(default_value=1, 
        help='number of waveforms in the Dazzler table').tag(config=True)

    nrepeat = traitlets.Int(default_value=1,
        help='number of times each waveform is repeated in camera file; not '\
        + 'the same as Dazzler NRepeat').tag(config=True)

    trigger_threshold = traitlets.Float(default_value=1.9, 
        help='voltage threshold for counting trigger as high')

    shutter_detect_pixel = traitlets.Int(default_value=670,
        help='pixel index to observe for shutter detection')

    shutter_transition_width = traitlets.Int(default_value=5, 
        help='number of laser shots around the shutter index to throw away '+\
            'as "partially open shots"')

    def __call__(self, dset):
        assert self.nrepeat == 1, "cannot handle nrepeat > 1"
        assert len(dset['achan1'].shape) == 1, "incorrect dset shape"

        table_starts, period = self.detect_table_start(dset['achan1'])
        
        # index the whole dataset simultaneously to keep only complete
        # table sets 
        dset = dset.isel(laser_shot=slice(table_starts[0], table_starts[-1]))
        
        nshots_probe_on, nshots_transition, nshots_probe_off = \
            self.detect_shutter_index(dset['spectrometer_data'])

        # fancy-ness: use a pandas MultiIndex to break laser shot index into 
        # a three-level index: t1 index, phase index, shutter
        shutter_levels = ['open', 'transition', 'closed']
        shutter_labels = np.hstack((
            np.zeros((nshots_probe_on, ), dtype=int),
            np.ones((nshots_transition,), dtype=int),
            2*np.ones((nshots_probe_off,), dtype=int)))

        assert shutter_labels.shape[0] == len(dset.coords['laser_shot']), \
                'label length must equal data length'

        # NOTE: to add nrepeat handling, reshape this to include nrepeat
        # dimension
        complete_indexes = table_starts[-1] - table_starts[0]

        tmp1 = np.empty((complete_indexes,), dtype=int)\
            .reshape(-1, self.nwaveforms, len(self.phase_cycles))
        tmp2 = np.empty((complete_indexes,), dtype=int)\
            .reshape(-1, self.nwaveforms, len(self.phase_cycles))

        for i, p in enumerate(self.phase_cycles):
            tmp1[:, :, i] = i
        
        for i in range(self.nwaveforms):
            tmp2[:, i, :] = i + self.nwaveforms*dset.attrs['tableidx']

        phase_labels = tmp1.ravel()
        t1_labels = tmp2.ravel()

        # voodoo logic to decide how many actual t1vals we have
        # assumes same number of waveforms in each table
        ntables = dset.attrs['maxidx'][1]
        t1_levels = list(range(self.nwaveforms*ntables))

        idx = pd.MultiIndex(levels=[t1_levels, self.phase_cycles, shutter_levels],
                labels=[t1_labels, phase_labels, shutter_labels],
                names=['t1_index', 'phase_cycle', 'shutter_label'])
        
        return dset.assign_coords(laser_shot=idx)

    def detect_table_start(self, array, waveform_repeat=1):
        '''detects peaks output when the Dazzler table starts at the begining by
        looking at the discrete difference of the array.
        '''

        diff = self.trigger_threshold 
        darr = np.diff(array)
        lohi, hilo = np.argwhere(darr > diff), np.argwhere(darr < -diff)

        # should be short cut evaluation, so second statement assumes same length
        # arrays
        if len(lohi) != len(hilo) or not np.all(lohi < hilo):
            #log.debug('spike showed up as the first or last signal')
            # check which one is longer, and which comes first 
            mlen = min(len(lohi), len(hilo))
            #log.debug('minimal length is {:d}'.format(mlen))
            if len(lohi) > mlen:
                # order is correct, but hilo is missing a point
                # truncate lohi at the end
                #log.debug('truncated lohi to match hilo')
                lohi = lohi[:-1]
                assert len(lohi) == len(hilo), 'did not help'
            elif len(hilo) > mlen:
                # order is incorrect, must be missing first point
                #log.debug('truncated hilo to match lohi')
                hilo = hilo[1:]
                assert len(lohi) == len(hilo), 'did not help'
            elif not np.all(lohi[:mlen] < hilo[:mlen]):
                #log.debug('both sides missing, shift over one')
                lohi, hilo = lohi[:-1], hilo[1:]
                assert len(lohi) == len(hilo), 'did not help'
            else:
                assert False, "should not happen!"

            mlen = max(len(lohi), len(hilo))
            #log.debug('truncating to {:d}'.format(mlen))
            lohi, hilo = lohi[:mlen], hilo[:mlen]

        if lohi.shape[0] == 1:
            s = 'only one complete dazzler table found. Are you sure' +\
                    ' you integrated for long enough?'
            #log.error(s)
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

            #log.error(s)
            raise RuntimeError(s)
            
        # diff returns forward difference. Add one for actual peak position 
        return np.squeeze(lohi)+1, period

    def detect_shutter_index(self, spectral_data):
        '''tries to determine a range in the camera frames that have the shutter on
        or off'''

        # throw away `transition_width` shots around the found position
        transition_width = self.shutter_transition_width


        avg_intensity = spectral_data.mean(dim='pixel')
        nshots = len(avg_intensity)
        shutter_start = int(np.abs(avg_intensity.diff(dim='laser_shot')).argmax())

        #print('Duty cycle', shutter_start/nshots)
        
        assert all([shutter_start != 0, shutter_start != nshots-1]), \
                'shutter start found at the beginning or the end of the trace'+\
                '. This is probably horribly wrong.'

        assert nshots > 2*transition_width, 'camera trace too short!'

        nshots_probe_on = int(shutter_start - transition_width)
        nshots_transition = int(2*transition_width)
        nshots_probe_off = int(nshots - (shutter_start + transition_width))
        return nshots_probe_on, nshots_transition, nshots_probe_off

def test():
    paths = ['/Volumes/Seagate Expansion Drive/stark-project-raw-data/16-11-01/rc-tgs-batch00',
             '/Volumes/Seagate Expansion Drive/stark-project-raw-data/16-11-01/rc-tgs-batch01']
    a = BatchLoader(paths[1])
    b = DataChunkLoader()
    b.trim_to = slice(10, None)
    c = PhaseCycler()
    c.phase_cycles = PhaseCycles['TGESS']
    c.nwaveforms = 1
    c.nrepeat = 1
    
    pool = Pool(8)
    pstarmap = toolz.curry(pool.starmap)
    pmap = toolz.curry(pool.map)
    
    pipeline = rcompose(a, curry(toolz.take)(1), pstarmap(b), map(c))
    res = list(pipeline())
    #print(res)
    
    import matplotlib.pyplot as plt

    print(res[0].sel(laser_shot={'shutter_label':'transition'}))
    
    tmp = res[0]['spectrometer_data'].sel(laser_shot={'shutter_label':'transition'})
    tmp2 = tmp.assign_coords(laser_shot=range(10))
    tmp2.plot()
    plt.show()
    


if __name__ == '__main__':
    #import timeit
    #print(timeit.timeit("test()", setup="from __main__ import test", number=1))

    test()

'''
class WorkerMetaClass(type):
    def __new__(mcs, name, bases, kwargs): 
        cls = super().__new__(mcs, name, bases, kwargs)
        
        # process inputs and outputs
        for inp in cls.inputs

        return cls

class Worker(metaclass=WorkerMetaClass):
    name = 'DemoWorker'
    inputs = [(NamedSignal('directory'), ]

    def new_directory(self, id):
        pass
'''
