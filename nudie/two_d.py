'''
# 2D ES analysis
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

def plot_phasing_tg(f, phasing_tg):
    import matplotlib.pylab as mpl
    mpl.plot(f, abs(phasing_tg))
    mpl.show()

def run(dd_name, dd_batch, when='today', wavelengths=None, plot=False,
        pump_chop=False,
        central_wl=None, phaselock_wl=None, pad_to=2048,
        waveforms_per_table=40, prd_est=850., lo_width=200, dc_width=200,
        gaussian_power=2.,
        analysis_path='./analyzed'):

    if plot:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.rcParams['figure.figsize'] = (16,12)
        mpl.use('Qt4Agg')
        plt.ioff()

    nrepeat = 1 # how many times each waveform is repeated in the camera file. Assumed to be one
    npixels = 1340
    trim_to = 3, -3

    # load up 2d data to use
    dd_info = next(nudie.load_job(job_name=dd_name, batch_set=[dd_batch], when=when))
    
    if pump_chop == True:
        phase_cycles = [(0., 0.), (1., 1.), (0, 0.6), (1., 1.6), (0, 1.3), (1., 2.3), 'chop']
    else:
        phase_cycles = [(0., 0.), (1., 1.), (0, 0.6), (1., 1.6), (0, 1.3), (1., 2.3)]

    # set current batch directory
    current_path = Path(dd_info['batch_path'])

    # generate hdf filename based on data date
    analysis_folder = Path(analysis_path) / dd_info['when']

    # create folder if it doesn't exist
    if not analysis_folder.exists():
        analysis_folder.mkdir(parents=True)

    save_path = analysis_folder / (dd_info['batch_name'] + '.h5')
    
    # remove data file if it exists
    if save_path.exists(): save_path.unlink()

    with h5py.File(str(save_path), 'w') as sf:
        # initialize groups
        sf.create_group('axes')

        shape = (dd_info['nt2'], dd_info['nt1'], npixels)
        sf.create_dataset('raw rephasing', shape, dtype=complex)
        sf.create_dataset('raw non-rephasing', shape, dtype=complex)
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
    
    ## Make phase-cycling coefficient matrix
    # subspaces
    if pump_chop == True:
        sub1 = np.array([[np.exp(1j*np.pi*(y - x)), np.exp(-1j*np.pi*(y - x)), 1] for x,y in phase_cycles[0:-1:2]])
        sub2 = np.array([[np.exp(1j*np.pi*(y - x)), np.exp(-1j*np.pi*(y - x)), 1] for x,y in phase_cycles[1:-1:2]])
    else:
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
            raise NotImplemented('Code is not setup to handle multiple loops')

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
        
        f, data, df = nudie.wavelen_to_freq(wl, data, ret_df=True, ax=0)
        
        # determine where the shutter is  
        shutter_open, shutter_closed, duty_cycle = nudie.determine_shutter_shots(data)
        
        # tag the phases
        shutter_info = {'last open idx': shutter_open, 'first closed idx': shutter_closed}                
        tags = nudie.tag_phases(start_idxs, period, tags=phase_cycles, nframes=data.shape[1], shutter_info=shutter_info)
        nudie.remove_incomplete_t1_waveforms(tags, phase_cycles)
        data_t = np.zeros((data.shape[1], data.shape[0]), dtype=float)
        
        # subtract the average of closed shutter shots from shutter open data
        for t1, k in it.product(range(waveforms_per_table), phase_cycles):
            idx_open = tags[nrepeat-1][t1][k]['shutter open']
            idx_closed = tags[nrepeat-1][t1][k]['shutter closed']
        
            data_t[idx_open, :] = data[:, idx_open].T \
                                  - data[:, idx_closed].mean(axis=1)

        t = np.fft.fftfreq(pad_to, df) 
        lo_window = gaussian2(lo_width, prd_est, t)
        dc_window = gaussian2(dc_width, 0, t)

        # fft everything
        fdata = np.fft.fft(data_t, axis=1, n=pad_to)

        if all([plot, table==0, t2==0]):
            tmp_fft = fdata[tags[0][0][phase_cycles[0]]['shutter open'][0], :]
            plot_windows(t, lo_window, dc_window, tmp_fft, prd_est)
        
        # spectral interferometry
        rIprobe = np.fft.ifft(fdata*dc_window, axis=1)[:, :npixels]
        rIlo = np.fft.ifft(fdata*lo_window, axis=1)[:, :npixels]
        
        if pump_chop:
            # Pump chop before division
            rEsig = np.zeros_like(rIlo)
            for t1, k in it.product(range(waveforms_per_table), phase_cycles[:-1]):
                phase_idx = tags[nrepeat-1][t1][k]['shutter open']
                chop_idx = tags[nrepeat-1][t1]['chop']['shutter open']

                # FIXME: not sure about sign. in rEsig, also whether to take
                # the square of rEprobe
                rEprobe = np.sqrt(np.abs(0.5*rIprobe[phase_idx, :] + 0.5*rIprobe[chop_idx, :]))
                rEsig[phase_idx, :] = -(rIlo[phase_idx, :] - rIlo[chop_idx, :])/rEprobe
        else:
            rEprobe = np.sqrt(np.abs(rIprobe))
            rEsig = rIlo/rEprobe
        
        # average each phase together
        if pump_chop:
            avg = np.zeros((len(phase_cycles[:-1]), waveforms_per_table, npixels), dtype=complex)
            for t1, (ik,k) in it.product(range(waveforms_per_table), enumerate(phase_cycles[:-1])):
                avg[ik, t1, :] = rEsig[tags[0][t1][k]['shutter open']].mean(axis=0)
        else:
            avg = np.zeros((len(phase_cycles), waveforms_per_table, npixels), dtype=complex)
            for t1, (ik,k) in it.product(range(waveforms_per_table), enumerate(phase_cycles)):
                avg[ik, t1, :] = rEsig[tags[0][t1][k]['shutter open']].mean(axis=0)
        
        r1, nr1, tg1, r2, nr2, tg2 = np.tensordot(Cperm_inv, avg, axes=[[1,],[0,]])
        R = 0.5*(r1 + r2)
        NR = 0.5*(nr1 + nr2)
        TG = 0.5*(tg1 + tg2)

        if all([plot, table==0, t2==0]):
            phasing_tg = 0.5*(R[0] + NR[0])
            plot_phasing_tg(f, phasing_tg)
        
        with h5py.File(str(save_path), 'a') as sf:
            # save data at current t2
            t1_slice = slice(table*waveforms_per_table,
                    (table+1)*waveforms_per_table)
            sf['raw rephasing'][t2, t1_slice] = R 
            sf['raw non-rephasing'][t2, t1_slice] = NR 
            sf['raw transient-grating'][t2, t1_slice] = TG 
            
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

        # add dimension scales
        rdata = sf['raw rephasing']
        rdata.dims.create_scale(gaxes['t2'], 'population time / fs')
        rdata.dims.create_scale(gaxes['t1'], 'pump delay time / fs')
        rdata.dims.create_scale(gaxes['detection frequency'], 'frequency / 1000 THz')
        rdata.dims.create_scale(gaxes['detection wavelength'], 'wavelength / nm')

        # attach them
        for tmp in [sf['raw rephasing'], sf['raw non-rephasing'], sf['raw transient-grating']]:
            tmp.dims[0].label = 'population time'
            tmp.dims[0].attach_scale(gaxes['t2'])
            tmp.dims[1].label = 'pump delay time'
            tmp.dims[1].attach_scale(gaxes['t1'])
            tmp.dims[2].label = 'detection axis'
            tmp.dims[2].attach_scale(gaxes['detection frequency'])
            tmp.dims[2].attach_scale(gaxes['detection wavelength'])

if __name__ == '__main__':
    from sys import argv

    # turn on printing of errors
    nudie.show_errors(nudie.logging.INFO)

    if len(argv) < 2:
        s = 'need a configuration file name as a parameter'
        nudie.error(s)
        raise RuntimeError(s)

    try:
        val = nudie.parse_config(argv[1])['2d']

        if val['stark']:
            s = 'the stark flag is set in the configuration. You should be ' +\
                'running the stark-2d.py script.'
            nudie.log.error(s)
            raise RuntimeError(s)

        run(dd_name=val['jobname'], 
                dd_batch=val['batch'], 
                when=val['when'],
                wavelengths=val['wavelengths'],
                plot=val['plot'],
                pump_chop=val['pump chop'],
                central_wl=val['central wl'],
                phaselock_wl=val['phaselock wl'],
                pad_to=val['zero pad to'],
                waveforms_per_table=val['waveforms per table'],
                prd_est=val['probe ref delay'], 
                lo_width=val['lo width'],
                dc_width=val['dc width'],
                gaussian_power=val['gaussian power'],
                analysis_path=val['analysis path'])
    except Exception as e:
        raise e
