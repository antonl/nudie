import logging
log = logging.getLogger(__name__)
import pathlib
import h5py
import re
import traitlets
import nudie
import itertools as it
from traitlets.config.configurable import Configurable
from traitlets.config import Config
from dask.delayed import delayed
from collections import namedtuple
from contextlib import contextmanager
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat

from .utils.batch_loader import load_job

axis = namedtuple('axis', ['label', 'dim', 'data'])

class RawData(Configurable):
    '''smallest unit of spectroscopy data that can be processed
    independently. Contains metadata, raw spectroscopy data, and other
    auxillary data. 

    Loaders create these objects from data on disk, and transformations 
    convert these objects to datasets of particular kinds.
    '''
    trim_to = traitlets.Instance(klass=slice, args=(10, None), 
        help='range of indexes to throw away when loading a camera file' +\
            '; used for removing saturated frames at the beginning of ' +\
            'acquisition', config=True)

    phase_cycles = traitlets.List(default_value=['frame'], minlen=1,
        help='iterable containing phase-cycle labels', config=True)

    nwaveforms = traitlets.Int(default_value=1, 
        help='number of waveforms in the Dazzler table', config=True)

    nrepeat = traitlets.Int(default_value=1,
        help='number of times each waveform is repeated in camera file; not '\
        + 'the same as Dazzler NRepeat', config=True)

    batch_name = traitlets.Unicode(default_value='test-batch', 
            help='name of job used in LabView software',
            config=True, allow_none=False)

    batch_path = traitlets.Unicode(help='location of the batch folder on disk', 
        config=True, allow_none=False)

    table = traitlets.Int(help='Dazzler table number', default_value=0,
            config=True)
    loop = traitlets.Int(help='camera loop number', default_value=0,
            config=True)
    t2 = traitlets.Int(help='T2 index', default_value=0, config=True)

    @traitlets.validate('batch_path')
    def _validate_batch_path(self, proposal):
        res = pathlib.Path(proposal['value'])
        if not res.exists():
            raise ValueError('batch path `{!s}` must exist'.format(res))

        return str(res.absolute())

    def __init__(self, **kwargs):
        super(RawData, self).__init__(config=Config(kwargs.pop('config', None)))

    def process(self):
        '''execute the loading of raw data from disk
        '''
        t2 = self.t2
        table = self.table
        loop = self.loop

        # first file requires special synchronization
        # this is the rule that determines that it is the first file
        first = 'first' if all([t2 == 0, table == 0, loop == 0]) else False
        
        # Load everything for the given t2, table, and loop value
        analogs = nudie.load_analogtxt(self.batch_name,
            self.batch_path, t2, table, loop)

        cdata = nudie.load_camera_file(self.batch_name,
            self.batch_path, t2, table, loop, force_uint16=True)
        
        # Synchronize it, trimming some frames in the beginning and end
        data, (a1, a2) = nudie.trim_all(*nudie.synchronize_daq_to_camera(\
                        cdata, analog_channels=analogs, which_file=first), 
                        trim_to=self.trim_to)

        data = data.astype(float) # convert to float64 before manipulating it!
        
        start_idxs, period = nudie.detect_table_start(a1)

        # determine where the shutter is
        shutter_open, shutter_closed, duty_cycle = nudie.determine_shutter_shots(data)
        shutter_info = {'last open idx': shutter_open, 'first closed idx': shutter_closed}                
        
        tags = nudie.tag_phases(start_idxs, period, tags=self.phase_cycles, nframes=data.shape[1], shutter_info=shutter_info)

        self.tags = tags
        self.camera_data = data

        achan = namedtuple('analog_channels', ['a1', 'a2'])
        self.analog_channels = achan(a1, a2)

        return self

class Transformation(Configurable):
    def __init__(self, *args, **kwargs):
        super(Configurable, self).__init__(self, *args, **kwargs)
        
    def apply(self, dset, other=[], **kwargs):
        return dset

class AttachWavelengthAxis(Transformation):
    def __init__(self, *args, **kwargs):
        super(Configurable, self).__init__(self, *args, **kwargs)
        
    def apply(self, dset, other=[], **kwargs):
        pass
 
class SpectralInterferometery(Transformation):
    def apply(self, dset, other=[]):
        pass

class Dataset(Configurable):
    type = 'Dataset'

    phase_cycles = ['frame']

    path = traitlets.Unicode(help='location of the dataset on disk', 
        config=True, allow_none=False)

    mode = traitlets.Enum(['r', 'r+', 'a', 'w', 'w-', 'x'], default_value='r', 
        config=True)

    allow_parallel = traitlets.Bool(default_value=False, config=True)    
    axes = traitlets.List(default_value=[])

    def __init__(self, *args, **kwargs):
        '''open hdf5 file representing spectroscopy data
        
        Parameters
        ----------
        path : str or pathlib.Path
            the location of the dataset
        mode : str
            file mode used to open the dataset ('r', 'r+', 'w', 'a', etc.)
        '''
        config = kwargs.pop('config', None)

        if config:
            config = Config(config)

        super(Dataset, self).__init__(config=config, **kwargs)
        new_mode = kwargs.pop('mode', None)
        if new_mode: 
            self.mode = new_mode
    
    def from_raw_data(self, iterable):
        return self

    def load(self, path=None):
        if path:
            self.path = path

        with self._raw_hdf5() as h5file:
            result = self._loadhook(h5file)
            
        return result

    def _loadhook(self, h5file):
        pass
    
    def save(self, path=None, overwrite=False):
        if path:
            self.path = path
        
        oldmode = self.mode
        self.mode = 'w' if overwrite else self.mode
        
        with self._raw_hdf5() as h5file:
            result = self._savehook(h5file)
            
        self.mode = oldmode
        return result
    
    def _savehook(self, h5file):
        '''hook in superclass that is called upon file save
        '''
        pass
    
    def compute(self):
        '''dummy method to allow use with or without parallel
        '''
        return self
    
    def transform(self, transform, *args, **kwargs):
        '''apply a transformation to the dataset to return a new dataset'''
        xfm = delayed(transform) if self.allow_parallel else transform
        return xfm(self, *args, **kwargs)
        
    @traitlets.validate('path')
    def _validate_batch_path(self, proposal):
        res = pathlib.Path(proposal['value'])
        return str(res.absolute())

    @contextmanager
    def _raw_hdf5(self):
        with h5py.File(self.path, self.mode) as sf:
            yield sf

class WavelengthCalibration(Dataset):
    phase_cycles = [] # unused
    
    calibration_coefficients = traitlets.Instance(klass=np.ndarray,
        help='coefficients of polynomial used to evaluate pixel values')
        
    calibration_type = traitlets.Enum(['polynomial fit', 'unknown'], 
        help='method used to get calibration')
    
    def __init__(self, *args, **kwargs):
        super(WavelengthCalibration, self).__init__(**kwargs)
        
        
    @classmethod
    def from_mat_file(klass, path):
        path = pathlib.Path(path)
        data = loadmat(str(path))
        inst = klass()        
        inst.calibrated_wavelength = np.squeeze(data['saved_wavelengths'])     
        px = np.arange(inst.calibrated_wavelength.shape[0], dtype=int)
        px_ax = axis(label='pixels', dim=0, data=px)
        inst.axes.append(px_ax)
        inst.path = str(path.with_suffix('.h5').absolute())
        inst.calibration_type = 'polynomial fit'
        inst.calibration_coefficients = np.squeeze(data['saved_calib_p'])
                
        return inst
    
    def _savehook(self, h5file):
        #gax = h5file.require_group('axes')
        data = h5file.require_dataset('calibrated wavelength', 
                            dtype=self.calibrated_wavelength.dtype,
                            shape=self.calibrated_wavelength.shape,
                            data=self.calibrated_wavelength)
        
        # save attributes
        for name, trait in self.class_own_traits().items():
            data.attrs[name] = trait.get(self)
        
        gax = h5file.require_group('axes')
        for ax in self.axes:                    
            fax = gax.require_dataset(ax.label, 
                                      shape=ax.data.shape, 
                                      dtype=ax.data.dtype,
                                      data=ax.data)
            data.dims[ax.dim].label = ax.label
            data.dims.create_scale(fax)
            data.dims[ax.dim].attach_scale(fax)
    
    def _loadhook(self, h5file):
        data = h5file['calibrated wavelength']
        
        for name, trait in self.class_own_traits().items():
            trait.set(self, data.attrs[name])
            
        self.calibrated_wavelength = np.array(data)
        
        gax = h5file.get('axes', default=None)
        if not gax:
            return # no axes to load
            
        for i,dim in enumerate(data.dims):
            if len(dim) < 1: # empty dim
                continue
            
            a = axis(label=dim.label, dim=i, data=np.array(dim[0]))
            if a.label not in [item.label for item in self.axes]:
                self.axes.append(a)        
    
class PP(Dataset):
    type = 'pump-probe'

    phase_cycles = ['none1', 'zero', 'none2', 'pipi']

    def __init__(self, *args, **kwargs):
        super(PP, self).__init__(*args, **kwargs)

class LinearStark(Dataset):
    type = 'linear Stark'

    phase_cycles = ['none1', 'zero', 'none2', 'pipi'] 

    field_on_threshold_volts = traitlets.Float(default_value=0.2,
        help='ADC value to consider the cutoff for field-on phases',
        config=True)

    dynamic_range = traitlets.Float(default_value=1e-7, 
        help='fraction of maximum value of reference intensity to mask, ' \
                + 'to avoid division by zero')
                
    linear_stark_meta = namedtuple('meta', 
        ['t2', 'table', 'loop', 'voltage_on', 'voltage_off'])
    
    def __init__(self, *args, **kwargs):
        super(LinearStark, self).__init__(*args, **kwargs)

    def from_raw_data(self, iterable):
        # define delayed functions if allowing parallel execution
        npw = delayed(np.where) if self.allow_parallel else np.where
        i1d = delayed(np.intersect1d) if self.allow_parallel else np.intersect1d
        ml = delayed(ma.masked_less) if self.allow_parallel else ma.masked_less
        l10 = delayed(np.log10) if self.allow_parallel else np.log10
        mf = delayed(ma.filled) if self.allow_parallel else ma.filled

        stark_spectra = []
        for rd in iterable:
            tags = rd.tags
            nrepeat = rd.nrepeat
            nwaveforms = rd.nwaveforms
            camera_data = delayed(rd.camera_data) if self.allow_parallel else rd.camera_data
            a2 = delayed(rd.analog_channels.a2) if self.allow_parallel else rd.analog_channels.a2 

            stark_idx = npw(a2 > self.field_on_threshold_volts)[0]
            nostark_idx = npw(a2 <= self.field_on_threshold_volts)[0]

            tmp = [] # holds stark spectra from different phases
            meta = [] # holds debug information for stark spectra (t2, table, loop)
            for i,k in enumerate(self.phase_cycles):
                # isolate indexes of scatter in different phases and remove
                so = i1d(tags[nrepeat-1][nwaveforms-1][k]['shutter open'], 
                         stark_idx, 
                         assume_unique=True)
                sc = i1d(tags[nrepeat-1][nwaveforms-1][k]['shutter closed'], 
                         stark_idx, 
                         assume_unique=True)
                no = i1d(tags[nrepeat-1][nwaveforms-1][k]['shutter open'], 
                         nostark_idx, 
                         assume_unique=True)
                nc = i1d(tags[nrepeat-1][nwaveforms-1][k]['shutter closed'], 
                         nostark_idx, 
                         assume_unique=True)                   
                            
                # remove probe scatter for stark shots
                field_on_frames = camera_data[:, so].mean(axis=-1) \
                        - camera_data[:, sc].mean(axis=-1)

                # and for no-stark shots
                field_off_frames = camera_data[:, no].mean(axis=-1) \
                    - camera_data[:, nc].mean(axis=-1)
                
                # mask small values to avoid division by zero
                field_off_frames_masked = ml(field_off_frames,
                        self.dynamic_range*field_off_frames.max())

                stark_spec = l10(field_on_frames/field_off_frames_masked)
                # fill masked areas with zero
                stark_spec_filled = mf(stark_spec, 0)
                tmp.append(stark_spec_filled)

            # average stark spectra from each phase cycle (they should be
            # identical
            if self.allow_parallel:
                avg_spectrum = delayed(sum)(tmp)/len(tmp)
                tmp_meta = delayed(self.linear_stark_meta)(rd.t2, rd.table, rd.loop, 
                    a2[stark_idx].mean(),
                    a2[nostark_idx].mean())
            else:
                avg_spectrum = sum(tmp)/len(tmp)
                tmp_meta = self.linear_stark_meta(rd.t2, rd.table, rd.loop, 
                    a2[stark_idx].mean(),
                    a2[nostark_idx].mean())

            stark_spectra.append(avg_spectrum)            
            meta.append(tmp_meta)

        if self.allow_parallel:
            self.stark_spectra = [s.compute() for s in stark_spectra]
            self.meta = [m.compute() for m in meta]
        else:
            self.stark_spectra = stark_spectra
            self.meta = meta
        
        # create voltage axis
        voltage_data = np.array([m.voltage_on - m.voltage_off for m in self.meta])
        voltage_axis = axis(label='voltage', dim=0, data=voltage_data)
        self.axes.append(voltage_axis)
        
        return self
    
    def _savehook(self, h5file):
        nspectra = len(self.stark_spectra)

        if nspectra < 1:
            log.debug('nothing to save')            
            return
        
        shape = self.stark_spectra[0].shape
        data = h5file.require_dataset('stark spectra', 
                shape=(nspectra, shape[0]), dtype=np.float64)
        
        for i,s in enumerate(self.stark_spectra):
            data[i] = s
            
        # save attributes
        for name, trait in self.class_own_traits().items():
            data.attrs[name] = trait.get(self)
        
        gax = h5file.require_group('axes')
        for ax in self.axes:                    
            fax = gax.require_dataset(ax.label, 
                                      shape=ax.data.shape, 
                                      dtype=ax.data.dtype,
                                      data=ax.data)
            data.dims[ax.dim].label = ax.label
            data.dims.create_scale(fax)
            data.dims[ax.dim].attach_scale(fax)
    
    def _loadhook(self, h5file):
        print('called loadhook')
        data = h5file['stark spectra']
        
        for name, trait in self.class_own_traits().items():
            trait.set(self, data.attrs[name])
            
        self.stark_spectra = np.array(data)
        
        gax = h5file.get('axes', default=None)
        if not gax:
            return # no axes to load
            
        for i,dim in enumerate(data.dims):
            if len(dim) < 1: # empty dim
                continue
            
            a = axis(label=dim.label, dim=i, data=np.array(dim[0]))
            if a.label not in [item.label for item in self.axes]:
                self.axes.append(a)
         
class DD(Dataset):
    type = '2DES'

    phase_cycles = [(0., 0.), (1., 1.), (0, 0.6), (1., 1.6), (0, 1.3), (1., 2.3)]

    def __init__(self, *args, **kwargs):
        super(DD, self).__init__(*args, **kwargs)

class DDESS(DD):
    type = '2DES Stark'

    def __init__(self, *args, **kwargs):
        super(DDESS, self).__init__(*args, **kwargs)

class TG(Dataset):
    type = 'transient-grating'

    phase_cycles = ['none1', 'zero', 'none2', 'pipi']

    def __init__(self, *args, **kwargs):
        super(TG, self).__init__(*args, **kwargs)

class TGStark(TG):
    type = 'transient-grating Stark'

    def __init__(self, *args, **kwargs):
        super(TGStark, self).__init__(*args, **kwargs)

class DatasetLoader(Configurable):
    type = 'NullLoader'

    def __init__(self, config=None):
        super(DatasetLoader, self).__init__(config=config)

        log.debug('created dataset loader `{!s}`'.format(self.type))

    def load(self, path, type, *args, **kwargs):
        # Load in setup specific way and return an iterable of RawData
        return []

class FastDazzlerSetupLoader(DatasetLoader):
    type = 'FastDazzler'

    trim_to = traitlets.Instance(klass=slice, args=(10, None), 
        help='range of indexes to throw away when loading a camera file' +\
            '; used for removing saturated frames at the beginning of ' +\
            'acquisition', config=True)

    batch_name = traitlets.Unicode(default_value='test-batch', 
            help='name of job used in LabView software',
            config=True, allow_none=False)

    batch_path = traitlets.Unicode(help='location of the batch folder on disk', 
        config=True, allow_none=False)

    nwaveforms = traitlets.Int(default_value=1, 
        help='number of waveforms in the Dazzler table', config=True)

    nrepeat = traitlets.Int(default_value=1,
        help='number of times each waveform is repeated in camera file; not '\
        + 'the same as Dazzler NRepeat', config=True)

    data_type = traitlets.Type(klass=Dataset, config=True)

    allow_parallel = traitlets.Bool(default_value=False, config=True)

    def __init__(self, config=None):
        super(FastDazzlerSetupLoader, self).__init__(config=config)

    def load(self):
        pattern = re.compile('(?P<job_name>[\w-]+)-batch(?P<batch_no>\d+)')

        p = pathlib.Path(self.batch_path)
        log.debug('loading {path!s}'.format(path=p.name))
        m = pattern.match(p.name)

        if m is None:
            raise ValueError('wrong job-name format')

        log.debug(
            'matched: job_name = `{job_name:s}`, batch_no = `{batch_no:s}`'.format(
                job_name=m.group('job_name'),
                batch_no=m.group('batch_no')))

        batch_info = next(load_job(m.group('job_name'), when=None,
                batch_set=set([int(m.group('batch_no'))]),
                data_path=p.parent))
        
        raw_datas = []
        # loop over available data
        for t2,table,loop in it.product(batch_info['t2_range'],
                batch_info['table_range'], batch_info['loop_range']):

            c = Config()

            # copy over settings used to RawData
            c.RawData.batch_name = self.batch_name
            c.RawData.batch_path = self.batch_path
            c.RawData.nwaveforms = self.nwaveforms
            c.RawData.nrepeat = self.nrepeat
            c.RawData.t2 = t2
            c.RawData.table = table
            c.RawData.loop = loop
            c.RawData.phase_cycles = self.data_type.phase_cycles

            c.Dataset.allow_parallel = self.allow_parallel
            c.merge(self.config)

            if self.allow_parallel:
                raw_datas.append(delayed(RawData)(config=c).process())
            else:
                raw_datas.append(RawData(config=c).process())

        # create dataset and process it
        if self.allow_parallel:
            dset = delayed(self.data_type)(config=c)
        else:
            dset = self.data_type(config=c)

        return dset.from_raw_data(raw_datas)
