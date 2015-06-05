'''
# Pump-Probe analysis

Analyze a pump-probe dataset for use in phasing 2D
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
import matplotlib.pylab as mpl
import pdb

def make_parser():
    parser = argparse.ArgumentParser(
        description='Analyze a single pump-probe dataset and output a hdf5 ' +\
                'file with the analysis.')
    parser.add_argument('config', type=str, action='store', 
        help='configuration file for running the analysis')
    return parser

def load_wavelengths(path):
    '''load a pre-calibrated wavengths file generated with the
    ```manual_calibration``` script in MATLAB.
    '''
    calib_file = Path(path)
    with calib_file.open('rb') as f:
        calib = loadmat(f)
        wl = np.squeeze(calib['saved_wavelengths'])
    return wl

def run(pp_name, pp_batch, when='today', wavelengths=None, plot=False,
        exclude=[], analysis_path='analyzed'):
    '''run the batch job to analyze pump-probe data
    
    Performs a shot-to-shot pump-probe analysis. 
    
    ```wavelengths``` is a path to a wavelengths file
    ```plot``` is a Boolean parameter controlling whether to show matplotlib
    plots
    '''
    # tags for phase cycles
    phase_cycles = ['none1', 'zero', 'none2', 'pipi']
    nrepeat = 1 # how many times each waveform is repeated in the camera file. Assumed to be one
    nwaveforms = 1 # number of dazzler waveforms in pump probe is 1
    npixels = 1340 # length of detection axis

    # load up pp data to use
    pp_info = next(nudie.load_job(job_name=pp_name, batch_set=[pp_batch], when=when))

    # set current batch directory
    current_path = Path(pp_info['batch_path'])

    # generate hdf filename based on data date
    analysis_folder = Path(analysis_path)

    # create folder if it doesn't exist
    if not analysis_folder.exists():
        analysis_folder.mkdir()

    save_path = analysis_folder / (pp_info['batch_name'] + '.h5')
    # remove data file if it exists
    if save_path.exists(): save_path.unlink()

    with h5py.File(str(save_path), mode='w') as sf:
        # initialize groups
        sf.create_group('axes')

        loops = np.array(list(filter(lambda x: x not in exclude, pp_info['loop_range'])), 
                dtype=int)
        shape = (pp_info['nt2'], len(loops), npixels)
        raw_pp = sf.create_dataset('raw pump-probe', shape, dtype=float)
        avg_pp = sf.create_dataset('averaged pump-probe', (shape[0], shape[2]), 
                dtype=float)
    try:
        wl = load_wavelengths(current_path.parent / wavelengths)
    except FileNotFoundError as e:
        nudie.log.error('Could not load wavelength calibration file!')
        raise e

    for t2,table,loop in it.product(pp_info['t2_range'], pp_info['table_range'],
            loops):
        
        # first file requires special synchronization
        # this is the rule that determines that it is the first file
        first = 'first' if all([t2 == 0, table == 0, loop == 0]) else False
        
        # Load everything for the given t2, table, and loop value
        analogs = nudie.load_analogtxt(pp_info['job_name'], current_path, t2, table, loop)
        cdata = nudie.load_camera_file(pp_info['job_name'], current_path, t2, table, loop, force_uint16=True)
        
        # Synchronize it, trimming some frames in the beginning and end
        data, (a1, a2) = nudie.trim_all(*nudie.synchronize_daq_to_camera(\
                        cdata, analog_channels=analogs, which_file=first))
        data = data.astype(float) # convert to float64 before manipulating it!
        
        # interpolate to even frequency spacing
        f, data, df = nudie.wavelen_to_freq(wl, data, ret_df=True, ax=0) 
        
        start_idxs, period = nudie.detect_table_start(a1)

        # determine where the shutter is
        shutter_open, shutter_closed, duty_cycle = nudie.determine_shutter_shots(data)
        
        # tag the phases
        shutter_info = {'last open idx': shutter_open, 'first closed idx': shutter_closed}                
        tags = nudie.tag_phases(start_idxs, period, tags=phase_cycles, 
                nframes=data.shape[1], shutter_info=shutter_info)

        # the first two indexes should be zero, so ignore them
        tags = tags[0][0]
        
        # truncate everything to the same length
        trunc_idx = slice(min([len(tags[k]['shutter open']) for k in phase_cycles]))
        
        # subtract scatter from shutter closed shots
        zero = ((data[:, tags['zero']['shutter open']][:, trunc_idx]).T \
                - data[:, tags['zero']['shutter closed']].mean(axis=-1)).T
        pipi = ((data[:, tags['pipi']['shutter open']][:, trunc_idx]).T \
                - data[:, tags['pipi']['shutter closed']].mean(axis=-1)).T
        none1 = ((data[:, tags['none1']['shutter open']][:, trunc_idx]).T \
                - data[:, tags['none1']['shutter closed']].mean(axis=-1)).T
        none2 = ((data[:, tags['none2']['shutter open']][:, trunc_idx]).T \
                - data[:, tags['none2']['shutter closed']].mean(axis=-1)).T
        
        # to match frank's code, except with minus signs
        A3 = (0.5*zero - 0.5*none1)
        B3 = (0.5*pipi - 0.5*none2)

        # FIXME: unsure about the proper signs for C3!
        C3 = (0.25*zero + 0.25*none1 + 0.25*pipi + 0.25*none2)
        S3 = np.mean((0.5*A3 + 0.5*B3)/np.sqrt(C3), axis=1)
        pp = S3/np.max(S3)
        
        with h5py.File(str(save_path), mode='a') as sf:
            loop_idx = np.argmin(np.abs(loops - loop))
            sf['raw pump-probe'][t2, loop_idx] = pp

    with h5py.File(str(save_path), mode='a') as sf:
        if plot:
            for t2, i in it.product(pp_info['t2_range'], range(len(loops))):
                mpl.plot(f, sf['raw pump-probe'][t2, i], 
                    label='t2 {:.1f} loop {:d}'.format(pp_info['t2'][t2][1],
                        loops[i]))
            mpl.legend()
            mpl.show()

        sf['averaged pump-probe'][:] = np.array(sf['raw pump-probe']).mean(axis=1)

        # write out meta data
        gaxes = sf['axes']
        gaxes['detection wavelength'] = nudie.spectrometer.speed_of_light/f
        gaxes['t2'] = pp_info['t2'][:, 1] # take only the time position 
        gaxes['loop'] = loops 
        gaxes['detection frequency'] = f 
        gaxes['detection frequency'].attrs['df'] = df

        sf.attrs['batch_name'] = pp_info['batch_name']
        sf.attrs['batch_no'] = pp_info['batch_no']
        sf.attrs['batch_path'] = pp_info['batch_path']
        sf.attrs['job_name'] = pp_info['job_name']
        sf.attrs['nt2'] = pp_info['nt2']
        sf.attrs['when'] = pp_info['when']

        sf.attrs['nloop'] = pp_info['loop_range'].stop
        sf.attrs['analysis timestamp'] = arrow.now().format('DD-MM-YYYY HH:mm')
        sf.attrs['nudie version'] = nudie.version

        # add dimension scales
        rpp = sf['raw pump-probe']
        rpp.dims.create_scale(gaxes['loop'], 'loop number')
        rpp.dims.create_scale(gaxes['t2'], 'population time / fs')
        rpp.dims.create_scale(gaxes['detection frequency'], 'detection frequency / 1000 THz')
        rpp.dims.create_scale(gaxes['detection wavelength'], 'detection wavelength / nm')
        
        rpp.dims[0].label = 'population time'
        rpp.dims[0].attach_scale(gaxes['t2'])
        rpp.dims[1].label = 'loop number'
        rpp.dims[1].attach_scale(gaxes['loop'])
        rpp.dims[2].label = 'detection axis'
        rpp.dims[2].attach_scale(gaxes['detection frequency'])
        rpp.dims[2].attach_scale(gaxes['detection wavelength'])

        app = sf['averaged pump-probe']
        app.dims[0].label = 'population time'
        app.dims[0].attach_scale(gaxes['t2'])
        app.dims[1].label = 'detection axis'
        app.dims[1].attach_scale(gaxes['detection frequency'])
        app.dims[1].attach_scale(gaxes['detection wavelength'])

if __name__ == '__main__':
    from sys import argv

    # turn on printing of errors
    nudie.show_errors(nudie.logging.INFO)

    if len(argv) < 2:
        s = 'need a configuration file name as a parameter'
        nudie.log.error(s)
        raise RuntimeError(s)

    try:
        try:
            val = nudie.parse_config(argv[1], which='pump probe')['pump probe']
        except ValueError as e:
            nudie.log.error('could not validate file. Please check ' +\
                'configuration options.')
            sys.exit(-1)
        
        run(pp_name=val['jobname'], pp_batch=val['batch'],
                when=val['when'], plot=val['plot'],
                wavelengths=val['wavelengths'],
                exclude=val['exclude'], analysis_path=val['analysis path'])
    except Exception as e:
        pass
