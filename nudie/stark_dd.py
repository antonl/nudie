'''
# Stark 2D ES analysis
'''

from __future__ import division, unicode_literals
import nudie
import h5py
import itertools as it
from scipy.signal import argrelmax, get_window
from scipy.optimize import minimize
from scipy.io import loadmat
from pathlib import Path
from collections import deque
import numpy as np
import arrow
import pdb
import matplotlib
import matplotlib.pyplot as mpl

matplotlib.rcParams['figure.figsize'] = (16,12)
matplotlib.use('Qt4Agg')

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
    idx = np.argmin(abs(t - prd_est))
    
    # we will only plot the absolute value of the FFT
    fft = np.abs(fft)/np.abs(fft[idx])
    
    mpl.plot(t, fft)
    mpl.plot(t, lo_window, linewidth=3)
    mpl.plot(t, dc_window, linewidth=3)
    mpl.ylim(-0.1, 1.1)
    mpl.show()

def plot_phasing_tg(f, phasing_tg):
    import matplotlib.pylab as mpl
    mpl.plot(f, abs(phasing_tg))
    mpl.show()

def run(dd_name, dd_batch, when='today', wavelengths=None, plot=False,
        central_wl=None, phaselock_wl=None, pad_to=2048,
        waveforms_per_table=40, prd_est=850., lo_width=200, dc_width=200, 
        gaussian_power=2,
        analysis_path='./analyzed', min_field=0.2):

    nrepeat = 1 # how many times each waveform is repeated in the camera file. Assumed to be one
    nstark = 2
    npixels = 1340
    trim_to = 3, -3

    # load up 2d data to use
    dd_info = next(nudie.load_job(job_name=dd_name, batch_set=[dd_batch], when=when))
    phase_cycles = [(0., 0.), (1., 1.), (0, 0.6), (1., 1.6), (0, 1.3), (1., 2.3)]

    # set current batch directory
    current_path = Path(dd_info['batch_path'])

    # generate hdf filename based on data date
    analysis_folder = Path(analysis_path)

    # create folder if it doesn't exist
    if not analysis_folder.exists():
        analysis_folder.mkdir()

    save_path = analysis_folder / (dd_info['batch_name'] + '.h5')
    
    # remove data file if it exists
    if save_path.exists(): save_path.unlink()

    with h5py.File(str(save_path), 'w') as sf:
        # initialize groups
        sf.create_group('axes')
        sf.attrs['nstark'] = nstark
        shape = (dd_info['nt2'], nstark, dd_info['nt1'], npixels)
        sf.create_dataset('raw rephasing', shape, dtype=complex)
        sf.create_dataset('raw non-rephasing', shape, dtype=complex)
        sf.create_dataset('raw transient-grating', shape, dtype=complex)

    try:
        wl = load_wavelengths(current_path.parent / wavelengths)
    except FileNotFoundError as e:
        nudie.log.error('Could not load wavelength calibration file!')
        raise e


    def gaussian2(w, x0, x):    
        c = 4*np.log(2)
        ξ = x-x0
        return np.exp(-c*(ξ/w)**(2*gaussian_power))

    ## Make phase-cycling coefficient matrix
    # subspaces
    sub1 = np.array([[np.exp(1j*np.pi*(y - x)), np.exp(-1j*np.pi*(y - x)), 1] for x,y in phase_cycles[0::2]])
    sub2 = np.array([[np.exp(1j*np.pi*(y - x)), np.exp(-1j*np.pi*(y - x)), 1] for x,y in phase_cycles[1::2]])

    perm = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1]])

    C = np.kron(np.array([[1, 0], [0, 0]]), sub1) + np.kron(np.array([[0, 0], [0, 1]]), sub2)
    Cinv = np.linalg.inv(C)
    Cperm_inv = np.dot(Cinv, perm.T)

    for loop, t2, table in it.product(dd_info['loop_range'], dd_info['t2_range'],
            dd_info['table_range']):

        if loop != 0:
            raise NotImplemented('code is not setup to handle multiple loops')

        # first file requires special synchronization
        # this is the rule that determines that it is the first file
        first = 'first' if all([t2 == 0, table == 0, loop == 0]) else False
        
        # Load everything for the given t2, table, and loop value
        analogs = nudie.load_analogtxt(dd_info['job_name'], current_path, t2, table, loop)
        cdata = nudie.load_camera_file(dd_info['job_name'], current_path, t2, table, loop, force_uint16=True)
        
        # Synchronize it, trimming some frames in the beginning and end
        data, (a1, a2) = nudie.trim_all(*nudie.synchronize_daq_to_camera(\
            cdata, analog_channels=analogs, which_file=first),
            trim_to=slice(*trim_to))
        data = data.astype(float) # convert to float64 before manipulating it!

        start_idxs, period = nudie.detect_table_start(a1)
        
        # data comes out [npixels, nframes]
        f, data, df = nudie.wavelen_to_freq(wl, data, ret_df=True, ax=0)
        
        # determine where the shutter is  
        shutter_open, shutter_closed, duty_cycle = nudie.determine_shutter_shots(data)
        
        # tag the phases
        shutter_info = {'last open idx': shutter_open, 'first closed idx': shutter_closed}                
        tags = nudie.tag_phases(start_idxs, period, tags=phase_cycles, 
            nframes=data.shape[1], shutter_info=shutter_info)

        # data_t is transposed data, [nframes, npixels]
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
            tmp_fft = fdata[tags[0][0][phase_cycles[0]]['shutter open'][0], :]
            plot_windows(t, lo_window, dc_window, tmp_fft, prd_est)
        
        # spectral interferometry
        rIprobe = np.fft.ifft(fdata*dc_window)[:, :npixels]
        rIlo = np.fft.ifft(fdata*lo_window)[:, :npixels]
        rEprobe = np.sqrt(np.abs(rIprobe))
        
        rEsig = rIlo/rEprobe
        
        # average each phase together
        stark_axis = np.array([a2[stark_idx].mean(), a2[nostark_idx].mean()])
        avg = np.zeros((len(phase_cycles), nstark, waveforms_per_table, npixels), dtype=complex)
        for t1, (ik,k) in it.product(range(waveforms_per_table), enumerate(phase_cycles)):
            # create indexes for stark shots
            idx_open = tags[nrepeat-1][t1][k]['shutter open']

            so = np.intersect1d(idx_open, stark_idx, assume_unique=True)
            no = np.intersect1d(idx_open, nostark_idx, assume_unique=True)

            avg[ik, 0, t1, :] = rEsig[so].mean(axis=0)
            avg[ik, 1, t1, :] = rEsig[no].mean(axis=0)
        
        r1, nr1, tg1, r2, nr2, tg2 = np.tensordot(Cperm_inv, avg, axes=[[1,],[0,]])
        R = 0.5*(r1 + r2)
        NR = 0.5*(nr1 + nr2)
        TG = 0.5*(tg1 + tg2)

        if all([plot, table==0, t2==0]):
            # plot the no-stark TG
            phasing_tg = 0.5*(R + NR)
            plot_phasing_tg(f, phasing_tg[1][0])
            plot_phasing_tg(f, phasing_tg[0][0])
            #mpl.contourf(np.abs(phasing_tg), 50)
            #mpl.show()
        
        with h5py.File(str(save_path), 'a') as sf:
            # save data at current t2
            t1_slice = slice(table*waveforms_per_table,
                    (table+1)*waveforms_per_table)
            sf['raw rephasing'][t2, :, t1_slice] = R 
            sf['raw non-rephasing'][t2, :, t1_slice] = NR 
            sf['raw transient-grating'][t2, :, t1_slice] = TG 
            
        del data, fdata, data_t, avg, rIprobe, rIlo, rEprobe, rEsig

    with h5py.File(str(save_path), 'a') as sf:
        # write out meta data
        sf.attrs['batch_name'] = dd_info['batch_name']
        sf.attrs['batch_no'] = dd_info['batch_no']
        sf.attrs['batch_path'] = dd_info['batch_path']
        sf.attrs['job_name'] = dd_info['job_name']
        sf.attrs['nt1'] = dd_info['nt1']
        sf.attrs['nt2'] = dd_info['nt2']
        sf.attrs['when'] = dd_info['when']

        sf.attrs['probe lo delay estimate'] = prd_est
        sf.attrs['analysis timestamp'] = arrow.now().format('DD-MM-YYYY HH:mm')
        sf.attrs['nudie version'] = nudie.version 
        
        # write out axes
        gaxes = sf.require_group('axes')
        freq_dataset = gaxes.create_dataset('detection frequency', data=f)
        freq_dataset.attrs['df'] = df
        gaxes.create_dataset('detection wavelength', data=nudie.spectrometer.speed_of_light/f)
        gaxes.create_dataset('t1', data=dd_info['t1'])
        gaxes.create_dataset('t2', data=dd_info['t2'][:, 1])
        gaxes.create_dataset('stark axis', data=stark_axis)

        # add dimension scales
        rdata = sf['raw rephasing']
        rdata.dims.create_scale(gaxes['t2'], 'population time / fs')
        rdata.dims.create_scale(gaxes['t1'], 'pump delay time / fs')
        rdata.dims.create_scale(gaxes['detection frequency'], 'frequency / 1000 THz')
        rdata.dims.create_scale(gaxes['detection wavelength'], 'wavelength / nm')
        rdata.dims.create_scale(gaxes['stark axis'], 'average field / kV')

        # attach them
        for tmp in [sf['raw rephasing'], sf['raw non-rephasing'], sf['raw transient-grating']]:
            tmp.dims[0].label = 'population time'
            tmp.dims[0].attach_scale(gaxes['t2'])
            tmp.dims[1].label = 'average voltage'
            tmp.dims[1].attach_scale(gaxes['stark axis'])
            tmp.dims[2].label = 'pump delay time'
            tmp.dims[2].attach_scale(gaxes['t1'])
            tmp.dims[3].label = 'detection axis'
            tmp.dims[3].attach_scale(gaxes['detection frequency'])
            tmp.dims[3].attach_scale(gaxes['detection wavelength'])

if __name__ == '__main__':
    from sys import argv

    # turn on printing of errors
    nudie.show_errors(nudie.logging.INFO)

    if len(argv) < 2:
        s = 'need a configuration file name as a parameter'
        nudie.log.error(s)
        raise RuntimeError(s)

    try:
        val = nudie.parse_config(argv[1])['2d']
    
        if val['stark'] != True:
            s = 'this is not a stark dataset according to the config file. ' +\
                    'Are you sure you shouldn\'t be running the 2d script?'
            nudie.log.error(s)
            raise RuntimeError(s)

        run(dd_name=val['jobname'], 
                dd_batch=val['batch'],
                when=val['when'], 
                wavelengths=val['wavelengths'],
                plot=val['plot'],
                central_wl=val['central wl'],
                phaselock_wl=val['phaselock wl'],
                pad_to=val['zero pad to'],
                waveforms_per_table=val['waveforms per table'],
                prd_est=val['probe ref delay'],
                lo_width=val['lo width'],
                dc_width=val['dc width'],
                gaussian_power=val['gaussian power'],
                analysis_path=val['analysis path'],
                min_field=val['field on threshold'])
    except Exception as e:
        raise e
