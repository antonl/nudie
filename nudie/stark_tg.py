'''
# TG  analysis
'''

from __future__ import division, unicode_literals
import nudie
import h5py
import itertools as it
from scipy.optimize import minimize
from scipy.io import loadmat
from pathlib import Path
from collections import deque
import numpy as np
import arrow
from scipy.signal import detrend
import numpy.ma as ma
import sys

def load_wavelengths(path):
    '''load a pre-calibrated wavengths file generated with the
    ```manual_calibration``` script in MATLAB.
    '''
    calib_file = Path(path)
    with calib_file.open('rb') as f:
        calib = loadmat(f)
        wl = np.squeeze(calib['saved_wavelengths'])
    return wl

def plot_windows(t, lo_window, dc_window, fft, prd_est):
    import matplotlib.pylab as mpl

    idx = np.argmin(abs(t - prd_est))
    
    # we will only plot the absolute value of the FFT
    fft = np.abs(fft)/np.abs(fft[idx])
    
    mpl.plot(t, fft)
    mpl.plot(t, lo_window, linewidth=3)
    mpl.plot(t, dc_window, linewidth=3)
    mpl.ylim(-0.1, 1.1)
    mpl.show()

def plot_phasing_tg(f, tg):
    import matplotlib.pylab as mpl
    mpl.plot(f, abs(tg))
    mpl.show()

def run(tg_name, tg_batch, when='today', wavelengths=None, plot=False, 
        pad_to=2048, prd_est=850., lo_width=200, dc_width=200,
        gaussian_power=2., analysis_path='./analyzed', min_field=0.2):

    if plot:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.rcParams['figure.figsize'] = (16,12)
        mpl.use('Qt4Agg')

    nrepeat = 1 # how many times each waveform is repeated in the camera file. Assumed to be one
    waveforms_per_table = 1
    npixels = 1340
    trim_to = 3, -3
    nstark = 2 # number of stark items

    # load up tg data to use
    tg_info = next(nudie.load_job(job_name=tg_name, batch_set=[tg_batch], when=when))
    
    phase_cycles = ['none1', 'zero', 'none2', 'pipi']

    # set current batch directory
    current_path = Path(tg_info['batch_path'])

    # generate hdf filename based on data date
    analysis_folder = Path(analysis_path)

    # create folder if it doesn't exist
    if not analysis_folder.exists():
        analysis_folder.mkdir(parents=True)

    save_path = analysis_folder / (tg_info['batch_name'] + '.h5')
    
    # remove data file if it exists
    if save_path.exists(): save_path.unlink()

    with h5py.File(str(save_path), 'w') as sf:
        # initialize groups
        sf.create_group('axes')
        sf.attrs['nstark'] = nstark
        shape = (tg_info['nt2'], nstark, npixels)
        sf.create_dataset('raw transient-grating', shape, dtype=complex)

    try:
        wl = load_wavelengths(current_path.parent / wavelengths)
    except FileNotFoundError as e:
        nudie.log.error('Could not load wavelength calibration file!')
        raise e

    # define gaussian
    def gaussian2(w, x0, x):    
        c = 4*np.log(2)
        ξ = x-x0
        return np.exp(-c*(ξ/w)**(2*gaussian_power))
    
    for loop, t2, table in it.product(tg_info['loop_range'], tg_info['t2_range'],
            tg_info['table_range']):

        if any([loop != 0, table != 0]):
            raise NotImplementedError('Code is not setup to handle multiple ' +\
                    'loops or tables.')

        # first file requires special synchronization
        # this is the rule that determines that it is the first file
        first = 'first' if all([t2 == 0, table == 0, loop == 0]) else False
        
        # Load everything for the given t2, table, and loop value
        analogs = nudie.load_analogtxt(tg_info['job_name'], current_path, t2, table, loop)
        cdata = nudie.load_camera_file(tg_info['job_name'], current_path, t2, table, loop, force_uint16=True)
        
        # Synchronize it, trimming some frames in the beginning and end
        data, (a1, a2) = nudie.trim_all(*nudie.synchronize_daq_to_camera(\
            cdata, analog_channels=analogs, which_file=first),
            trim_to=slice(*trim_to))
        data = data.astype(float) # convert to float64 before manipulating it!

        start_idxs, period = nudie.detect_table_start(a1)
        
        f, data, df = nudie.wavelen_to_freq(wl, data, ret_df=True, ax=0)
        
        # determine where the shutter is  
        shutter_open, shutter_closed, duty_cycle = nudie.determine_shutter_shots(data)
        
        # tag the phases
        shutter_info = {'last open idx': shutter_open, 'first closed idx': shutter_closed}                
        tags = nudie.tag_phases(start_idxs, period, tags=phase_cycles, nframes=data.shape[1], shutter_info=shutter_info)
        nudie.remove_incomplete_t1_waveforms(tags, phase_cycles)
        data_t = np.zeros((data.shape[1], data.shape[0]), dtype=float)

        # verify synchronization of applied field to Dazzler
        for t1, k in it.product(range(waveforms_per_table), phase_cycles):
            # check that every second frame in analog channel 2 is either high field, or low field, but not both
            idx_open = tags[nrepeat-1][t1][k]['shutter open']
            idx_closed = tags[nrepeat-1][t1][k]['shutter closed']

            # Offset 0, every second frame has high field
            ao = a2[idx_open][0::2] > min_field
            # Offset 0, every second frame has low field
            bo = a2[idx_open][0::2] <= min_field 
            
            # Offset 1, every second frame has high field
            co = a2[idx_open][1::2] > min_field
            # Offset 1, every second frame has low field
            do = a2[idx_open][1::2] <= min_field
            
            # Offset 0, every second frame has high field
            ac = a2[idx_closed][0::2] > min_field
            # Offset 0, every second frame has low field
            bc = a2[idx_closed][0::2] <= min_field 
            
            # Offset 1, every second frame has high field
            cc = a2[idx_closed][1::2] > min_field
            # Offset 1, every second frame has low field
            dc = a2[idx_closed][1::2] <= min_field
            
            assert np.logical_xor(np.all(ao), np.all(bo)), \
                'detected stark synchronization error, offset 0, phase %s' % str(k)
            assert np.logical_xor(np.all(co), np.all(do)), \
                'detected stark synchronization error, offset 1, phase %s' % str(k)
            
            assert np.logical_xor(np.all(ac), np.all(bc)), \
                'detected stark synchronization error, offset 0, phase %s' % str(k)
            assert np.logical_xor(np.all(cc), np.all(dc)), \
                'detected stark synchronization error, offset 1, phase %s' % str(k)

            assert np.logical_xor(np.all(ao), np.all(co)), \
                'detected stark synchronization error, offset 0, phase %s' % str(k)
            assert np.logical_xor(np.all(bo), np.all(do)), \
                'detected stark synchronization error, offset 1, phase %s' % str(k)

        # determine stark-on indexes
        stark_idx = np.where(a2 > min_field)[0]
        nostark_idx = np.where(a2 <= min_field)[0]

        # subtract the average of closed shutter shots from shutter open data
        for t1, k in it.product(range(waveforms_per_table), phase_cycles):
            idx_open = tags[nrepeat-1][t1][k]['shutter open']
            idx_closed = tags[nrepeat-1][t1][k]['shutter closed']

            so = np.intersect1d(idx_open, stark_idx, assume_unique=True)
            sc = np.intersect1d(idx_closed, stark_idx, assume_unique=True)                   

            no = np.intersect1d(idx_open, nostark_idx, assume_unique=True)
            nc = np.intersect1d(idx_closed, nostark_idx, assume_unique=True)                   

            data_t[so, :] = (data[:, so].T - data[:, sc].mean(axis=1))
            data_t[no, :] = (data[:, no].T - data[:, nc].mean(axis=1))
        
        t = np.fft.fftfreq(pad_to, df) 
        lo_window = gaussian2(lo_width, prd_est, t)
        dc_window = gaussian2(dc_width, 0, t)

        # fft everything
        fdata = np.fft.fft(data_t, axis=1, n=pad_to)

        if all([plot, table==0, t2==0]):
            tmp_fft = fdata[tags[0][0][phase_cycles[1]]['shutter open'][0], :]
            plot_windows(t, lo_window, dc_window, tmp_fft, prd_est)
        
        # spectral interferometry
        rIprobe = np.fft.ifft(fdata*dc_window, axis=1)[:, :npixels]
        rIlo = np.fft.ifft(fdata*lo_window, axis=1)[:, :npixels]
        
        rEprobe = np.sqrt(np.abs(rIprobe))

        '''
        tag_none1 = tags[0][0]['none1']['shutter open']
        nnone1 = rEprobe[tag_none1].mean(axis=0)
        none1 = rIlo[tag_none1].mean(axis=0)
        tag_zero = tags[0][0]['zero']['shutter open']
        nzero = rEprobe[tag_zero].mean(axis=0)
        zero = rIlo[tag_zero].mean(axis=0)
        tag_none2 = tags[0][0]['none2']['shutter open']
        nnone2 = rEprobe[tag_none2].mean(axis=0)
        none2 = rIlo[tag_none2].mean(axis=0)
        tag_pipi = tags[0][0]['pipi']['shutter open']
        npipi = rEprobe[tag_pipi].mean(axis=0)
        pipi = rIlo[tag_pipi].mean(axis=0)
        '''
        stark_tag = lambda x: np.intersect1d(x, stark_idx, assume_unique=True)
        nostark_tag = lambda x: np.intersect1d(x, nostark_idx, assume_unique=True)

        tag_none1 = tags[0][0]['none1']['shutter open']
        stag_none1 = stark_tag(tag_none1)
        ntag_none1 = nostark_tag(tag_none1)

        #snone1 = rEprobe[stag_none1]
        #nnone1 = rEprobe[ntag_none1]
        snone1 = rIlo[stag_none1].mean(axis=0)
        nnone1 = rIlo[ntag_none1].mean(axis=0)

        tag_zero = tags[0][0]['zero']['shutter open']
        stag_zero = stark_tag(tag_zero)
        ntag_zero = nostark_tag(tag_zero)
        snzero = rEprobe[stag_zero].mean(axis=0)
        nnzero = rEprobe[ntag_zero].mean(axis=0)
        szero = rIlo[stag_zero].mean(axis=0)
        nzero = rIlo[ntag_zero].mean(axis=0)

        tag_none2 = tags[0][0]['none2']['shutter open']
        stag_none2 = stark_tag(tag_none2)
        ntag_none2 = nostark_tag(tag_none2)

        #nnone2 = rEprobe[tag_none2]
        snone2 = rIlo[stag_none2].mean(axis=0)
        nnone2 = rIlo[ntag_none2].mean(axis=0)

        tag_pipi = tags[0][0]['pipi']['shutter open']
        stag_pipi = stark_tag(tag_pipi)
        ntag_pipi = nostark_tag(tag_pipi)
        snpipi = rEprobe[stag_pipi].mean(axis=0)
        nnpipi = rEprobe[ntag_pipi].mean(axis=0)
        spipi = rIlo[stag_pipi].mean(axis=0)
        npipi = rIlo[ntag_pipi].mean(axis=0)

        #TG = 1/(nzero + npipi)*(zero/nzero + pipi/npipi) \
        #        - (1/(nnone1 + nnone2)*(none1/nnone1 + none2/nnone2))
        #TG = ((zero + pipi - none1 - none2)/(nzero + npipi)).mean(axis=0)

        stark_axis = np.array([a2[stark_idx].mean(), a2[nostark_idx].mean()])
        #TG = ((zero + pipi - none1 - none2)/(nzero + npipi))
        sTG = ((szero + spipi - snone1 - snone2)/(snzero + snpipi))
        nTG = ((nzero + npipi - nnone1 - nnone2)/(nnzero + nnpipi))

        '''
        if all([plot, table==0, t2==0]):
            plot_phasing_tg(f, TG)
            plot_phasing_tg(f, (nzero + npipi).mean(axis=0))
        '''

        with h5py.File(str(save_path), 'a') as sf:
            # save data at current t2
            
            # for some reason, I need -1j now -> 1j to make the real part the
            # absorptive and -1 to do the 2D convention
            sf['raw transient-grating'][t2, 0] = -1j*sTG 
            sf['raw transient-grating'][t2, 1] = -1j*nTG

    with h5py.File(str(save_path), 'a') as sf:
        # write out meta data
        sf.attrs['batch_name'] = tg_info['batch_name']
        sf.attrs['batch_no'] = tg_info['batch_no']
        sf.attrs['batch_path'] = tg_info['batch_path']
        sf.attrs['job_name'] = tg_info['job_name']
        sf.attrs['nt2'] = tg_info['nt2']
        sf.attrs['when'] = tg_info['when']

        sf.attrs['detection axis pad to'] = pad_to
        sf.attrs['probe lo delay estimate'] = prd_est
        sf.attrs['analysis timestamp'] = arrow.now().format('DD-MM-YYYY HH:mm')
        sf.attrs['nudie version'] = nudie.version
        sf.attrs['experiment type'] = 'stark transient-grating'
        
        # write out axes
        gaxes = sf.require_group('axes')
        gaxes.create_dataset('stark axis', data=stark_axis)
        freq_dataset = gaxes.create_dataset('detection frequency', data=f)
        freq_dataset.attrs['df'] = df
        gaxes.create_dataset('detection wavelength', data=nudie.spectrometer.speed_of_light/f)
        gaxes.create_dataset('t2', data=tg_info['t2'][:, 1])

        # add dimension scales and attach them
        rdata = sf['raw transient-grating']
        rdata.dims.create_scale(gaxes['t2'], 'population time / fs')
        rdata.dims[0].label = 'population time'
        rdata.dims[0].attach_scale(gaxes['t2'])
        rdata.dims.create_scale(gaxes['stark axis'], 'average field / kV')
        rdata.dims[1].label = 'average voltage'
        rdata.dims[1].attach_scale(gaxes['stark axis'])
        rdata.dims.create_scale(gaxes['detection frequency'], 'frequency / 1000 THz')
        rdata.dims.create_scale(gaxes['detection wavelength'], 'wavelength / nm')
        rdata.dims[2].label = 'detection axis'
        rdata.dims[2].attach_scale(gaxes['detection frequency'])
        rdata.dims[2].attach_scale(gaxes['detection wavelength'])

def main(config, verbosity=nudie.logging.INFO):
    nudie.show_errors(verbosity)

    try:
        try:
            val = nudie.parse_config(config, which='tg')['tg']
        except ValueError as e:
            nudie.log.error('could not validate file. Please check ' +\
                'configuration options.')
            return

        if val['stark'] != True:
            s = 'the stark flag is not set in the configuration. ' +\
                'Are you sure you shouldn\'t be running the tg script?'
            nudie.log.error(s)
            return

        run(tg_name=val['jobname'], 
            tg_batch=val['batch'], 
            when=val['when'],
            wavelengths=val['wavelengths'],
            plot=val['plot'],
            pad_to=val['detection axis zero pad to'],
            prd_est=val['probe ref delay'], 
            lo_width=val['lo width'],
            dc_width=val['dc width'],
            gaussian_power=val['gaussian power'],
            analysis_path=val['analysis path'],
            min_field=val['field on threshold'],
            datapath=val['data path'])
    except Exception as e:
        nudie.log.exception(e)

if __name__ == '__main__':
    from sys import argv

    # turn on printing of errors
    level = nudie.logging.INFO
    nudie.show_errors(level)

    if len(argv) < 2:
        s = 'need a configuration file name as a parameter'
        nudie.log.error(s)
        sys.exit(-1)

    main(argv[1], level)

