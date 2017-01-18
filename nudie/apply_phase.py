'''
# Phasing of 2D data to a pump probe spectrum

'''

from __future__ import division, unicode_literals
import nudie
import h5py
import itertools as it
from scipy.signal import argrelmax, get_window
from scipy.optimize import minimize, basinhopping, leastsq
from pathlib import Path
from collections import deque
import numpy as np
import arrow
import matplotlib.pylab as mpl
import sys
import pdb

def plot_phased_tg(t2, f3, tg, stark=False):
    from nudie.utils.plotting import shiftedColorMap

    if stark:
        return

    res = np.real(tg).T

    nlevels = 50
    vmax, vmin = np.max(res), np.min(res)
    ticker = mpl.mpl.ticker.MaxNLocator(nlevels)
    levels = ticker.tick_values(vmin, vmax)
    #levels = levels[np.abs(levels) > levels_threshold] 
    xsection = res[800, :]

    midpoint = 1-vmax/(vmax+abs(vmin))
    cmap = shiftedColorMap(mpl.cm.RdBu_r, midpoint=midpoint)
        
    fig = mpl.figure()
    gs = mpl.GridSpec(2,2, height_ratios=[2,1], width_ratios=[5, 0.1]) 
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    ax1.contour(-t2, f3, res, 10, colors='k')
    cf = ax1.contourf(-t2, f3, res, levels=levels, cmap=cmap)
        
    ax1.set_ylabel('detection frequency')
    ax1.set_xlabel('t2 time')            
    mpl.colorbar(cf, cax=ax3, use_gridspec=True)

    ax2.plot(-t2, xsection)
    ax2.grid()
    gs.tight_layout(fig)
    mpl.show()

def apply_phase_tg(tg_file, correction_multiplier, correction_offset, **kwargs):
    nudie.log.debug('apply_phase_tg got kwargs: {!s}'.format(kwargs))

    # unpack needed kwargs
    force = kwargs.get('force', False)
    stark = kwargs.get('stark', False)
    plot = kwargs.get('plot', False)

    with h5py.File(str(tg_file), 'a') as sf:
        # check that file is not already phased
        if 'phased' in sf.attrs:
            if not force:
                s = 'Dataset already has a phasing correction. ' +\
                    'Use the force flag to overwrite.'
                nudie.log.warning(s)
                raise RuntimeError(s)

        sf.require_dataset('correction multiplier',
                shape=correction_multiplier.shape,
                dtype=complex, data=correction_multiplier) 
        sf.require_dataset('correction offset', shape=(), dtype=float,
                data=correction_offset)
        sf.attrs['copied phase'] = True

        TG = sf['raw transient-grating']

        if stark:
            print('TG shape: ' + str(TG.shape))
            nstark = 2
            spectra_shape = (sf.attrs['nt2'], nstark, TG.shape[2])
            phased_TG = correction_multiplier*TG + correction_offset
            tg = sf.require_dataset('phased transient-grating', shape=spectra_shape,
                    dtype=complex, data=phased_TG) 
        else:
            spectra_shape = (sf.attrs['nt2'], TG.shape[1])

            # is TG scaled the same way as the 2D? I don't think so...
            phased_TG = correction_multiplier*TG + correction_offset
            tg = sf.require_dataset('phased transient-grating', shape=spectra_shape,
                    dtype=complex, data=phased_TG) 

        for x in [tg,]:
            if stark:
                x.dims[0].label = 'population time'
                x.dims[0].attach_scale(sf['axes/t2'])
                x.dims[1].label = 'average voltage'
                x.dims[1].attach_scale(sf['axes/stark axis'])
                x.dims[2].label = 'detection frequency'
                x.dims[2].attach_scale(sf['axes/detection frequency'])
            else:
                x.dims[0].label = 'population time'
                x.dims[0].attach_scale(sf['axes/t2'])
                x.dims[1].label = 'detection frequency'
                x.dims[1].attach_scale(sf['axes/detection frequency'])

        if plot:
            t2_ax = sf['axes/t2'][:]
            f3_ax = sf['axes/detection frequency'][:]
            if stark:
                plot_phased_tg(t2_ax, f3_ax, phased_TG, stark=True)
            else:
                plot_phased_tg(t2_ax, f3_ax, phased_TG)

def apply_phase_2d(dd_file, correction_multiplier, correction_offset, **kwargs):
    nudie.log.debug('apply_phase_2d got kwargs: {!s}'.format(kwargs))

    # unpack needed kwargs
    phaselock_wl = kwargs['phaselock_wl']
    central_wl = kwargs['central_wl']
    excitation_axis_pad_to = kwargs['excitation_axis_pad_to']
    force = kwargs.get('force', False)
    stark = kwargs.get('stark', False)
    plot = kwargs.get('plot', False)

    # load 2D data
    with h5py.File(str(dd_file), 'a') as sf:
        # check that file is not already phased
        if 'phased' in sf.attrs:
            if not force:
                s = 'Dataset already has a phasing correction. ' +\
                    'Use the force flag to overwrite.'
                nudie.log.warning(s)
                raise RuntimeError(s)

        sf.require_dataset('correction multiplier',
                shape=correction_multiplier.shape,
                dtype=complex, data=correction_multiplier) 
        sf.require_dataset('correction offset', shape=(), dtype=float,
                data=correction_offset)
        sf.attrs['copied phase'] = True

        R = sf['raw rephasing']
        NR = sf['raw non-rephasing']

        if stark:
            dd_f = np.array(R.dims[3][0])
            dd_t1 = np.array(R.dims[2][0])
            dd_stark = np.array(R.dims[1][0])
            dd_t2 = np.array(R.dims[0][0])
        else:
            dd_f = np.array(R.dims[2][0])
            dd_t1 = np.array(R.dims[1][0])
            dd_t2 = np.array(R.dims[0][0])

        R = np.array(R)
        NR = np.array(NR)

    # fft R and NR, deleting temporaries so that we don't store too much stuff
    # in memory

    with h5py.File(str(dd_file), 'a') as sf:
        # create excitation axis
        t1 = sf['axes/t1']
        dt1 = abs(t1[1] - t1[0])
        f1 = np.fft.fftshift(np.fft.fftfreq(excitation_axis_pad_to, dt1))
        
        f1_ax = sf['axes'].require_dataset('raw excitation frequency', 
            shape=f1.shape, dtype=float)
        f1_ax[:] = f1

        # trying frank's method of phase-locking axis FIXME!!!!
        # this is due to a sign error in the Dazzler code 
        # Note: This is now fixed. Bug was due to one extra fftshift
        C = nudie.spectrometer.speed_of_light
        #f1_pl = f1 + C/(2*central_wl - phaselock_wl)
        f1_pl = f1 + C/phaselock_wl

        f1_ax_pl = sf['axes'].require_dataset('phase-locked excitation frequency',
            shape=f1.shape, dtype=float)
        f1_ax_pl[:] = f1_pl 

        if stark:
            spectra_shape = (sf.attrs['nt2'], sf.attrs['nstark'], f1.shape[0], dd_f.shape[0])
        else:
            spectra_shape = (sf.attrs['nt2'], f1.shape[0], dd_f.shape[0])

        # zero pad data and flip axes
        # properly combine the rephasing and nonrephasing
        roll_by = 1 if excitation_axis_pad_to % 2 == 0 else 0

        t1_len = t1.shape[0]
        window_sym = get_window(('kaiser', 8.0), 2*t1_len, fftbins=True)
        window_sym[t1_len] = 0.5
        window = window_sym[t1_len:]

        if stark:
            window_func = lambda x: np.einsum('ijkl,k->ijkl', x, window)
            diagnostic_R = window_func(R)
            diagnostic_NR = window_func(NR)
            Rw1 = np.fft.fftshift(np.fft.fft(diagnostic_R, axis=2, n=excitation_axis_pad_to), axes=2)
            NRw1 = np.fft.fftshift(np.fft.fft(diagnostic_NR, axis=2, n=excitation_axis_pad_to), axes=2)

            Sw1 = 0.5*(np.roll(Rw1[:, :, ::-1], shift=roll_by, axis=2) + NRw1)
        else:
            window_func = lambda x: np.einsum('ijk,j->ijk', x, window)
            diagnostic_R = window_func(R)
            diagnostic_NR = window_func(NR)
            Rw1 = np.fft.fftshift(np.fft.fft(diagnostic_R, axis=1, n=excitation_axis_pad_to), axes=1)
            NRw1 = np.fft.fftshift(np.fft.fft(diagnostic_NR, axis=1, n=excitation_axis_pad_to), axes=1)
            Sw1 = 0.5*(np.roll(Rw1[:, ::-1], shift=roll_by, axis=1) + NRw1)

        phased_Rw1 = correction_multiplier*Rw1 + correction_offset
        r = sf.require_dataset('phased rephasing', shape=spectra_shape,
                dtype=complex, data=phased_Rw1) 

        phased_NRw1 = correction_multiplier*NRw1 + correction_offset
        nr = sf.require_dataset('phased non-rephasing', shape=spectra_shape,
                dtype=complex, data=phased_NRw1) 

        phased_2D = correction_multiplier*Sw1 + correction_offset
        dd = sf.require_dataset('phased 2D', shape=spectra_shape, 
                dtype=complex, data=phased_2D)

        # add diagnostic rephasing and nonrephasing dsets
        sf.require_dataset('windowed raw rephasing', shape=R.shape,
                               dtype=complex, data=correction_multiplier*diagnostic_R + correction_offset)
        sf.require_dataset('windowed raw non-rephasing', shape=NR.shape,
                               dtype=complex, data=correction_multiplier*diagnostic_NR + correction_offset)
        sf.require_dataset('phased raw rephasing', shape=R.shape,
                           dtype=complex, data=correction_multiplier*R + correction_offset)
        sf.require_dataset('phased raw non-rephasing', shape=NR.shape,
                           dtype=complex, data=correction_multiplier*NR + correction_offset)

        sf.attrs['phase lock wavelength'] = phaselock_wl 
        sf.attrs['central wavelength'] = central_wl 
        # attach dimension scales
        r.dims.create_scale(f1_ax, 'raw excitation frequency')
        r.dims.create_scale(f1_ax_pl, 'phase-locked excitation frequency')

        for x in [r, nr, dd]:
            if stark:
                x.dims[0].label = 'population time'
                x.dims[0].attach_scale(sf['axes/t2'])
                x.dims[1].label = 'average voltage'
                x.dims[1].attach_scale(sf['axes/stark axis'])
                x.dims[2].label = 'pump frequency'
                x.dims[2].attach_scale(f1_ax)
                x.dims[2].attach_scale(f1_ax_pl)
                x.dims[3].label = 'detection frequency'
                x.dims[3].attach_scale(sf['axes/detection frequency'])
            else:
                x.dims[0].label = 'population time'
                x.dims[0].attach_scale(sf['axes/t2'])
                x.dims[1].label = 'pump frequency'
                x.dims[1].attach_scale(f1_ax)
                x.dims[1].attach_scale(f1_ax_pl)
                x.dims[2].label = 'detection frequency'
                x.dims[2].attach_scale(sf['axes/detection frequency'])

experiment_map = {
    'transient-grating': (apply_phase_tg, {}), 
    'stark transient-grating': (apply_phase_tg, {'stark':True}), 
    '2d': (apply_phase_2d, {}),
    'stark 2d': (apply_phase_2d, {'stark':True}),
    }

def run(path, ref_name, ref_batch, exp_name, exp_batch, plot=False,
        force=False, stark=False):
    path = Path(path)

    # create folder if it doesn't exist
    if not path.exists():
        s = 'Path {!s} doesn\'t exist. Please correct.'.format(path)
        nudie.log.error(s)
        raise FileNotFoundError(s)

    ref_file = path / '{:s}-batch{:02d}.h5'.format(ref_name, ref_batch)
    exp_file = path / '{:s}-batch{:02d}.h5'.format(exp_name, exp_batch)
    
    if not all([ref_file.exists(), exp_file.exists()]):
        s = 'Could not find one of the data files. ' +\
            'Please check that `{!s}` and `{!s}` exist.'.format(ref_file, exp_file)
        nudie.log.error(s)
        raise FileNotFoundError(s)

    # load phased data
    with h5py.File(str(ref_file), 'r') as sf:
        if sf.attrs.get('phased', False) == False:
            s = 'phasing dataset does not contain a phasing correction. ' +\
                'Cannot reuse it for phasing!'
            nudie.log.error(s)
            raise RuntimeError(s)

        phaselock_wl = sf.attrs['phase lock wavelength'] 
        central_wl = sf.attrs['central wavelength'] 
        excitation_axis_pad_to = sf.attrs['excitation axis zero pad to']
        correction_multiplier = np.array(sf['correction multiplier'])
        correction_offset = np.array(sf['correction offset'])

    with h5py.File(str(exp_file), 'r') as sf:
        exp_type = sf.attrs['experiment type']
    
    exp_func, exp_kwargs = experiment_map[exp_type]

    # dispatch to apply_phase function
    exp_kwargs['plot'] = plot
    exp_kwargs['force'] = force
    exp_kwargs['central_wl'] = central_wl
    exp_kwargs['phaselock_wl'] = phaselock_wl
    exp_kwargs['excitation_axis_pad_to'] = excitation_axis_pad_to

    exp_func(exp_file, correction_multiplier, correction_offset, **exp_kwargs)

    with h5py.File(str(exp_file), 'a') as sf:
        # attach phased exp attribute
        sf.attrs['phased'] = True
        sf.attrs['phasing timestamp'] = arrow.now().format('DD-MM-YYYY HH:MM')
        sf.attrs['nudie version'] = nudie.version

def main(config, verbosity=nudie.logging.INFO):
    nudie.show_errors(verbosity)

    try:
        try:
            val = nudie.parse_config(config, which='phasing')['phasing']
        except ValueError as e:
            nudie.log.error('could not validate file. Please check ' +\
                'configuration options.')
            return

        if val['copy'] == False:
            s = '`copy` flag is not set. You should be using the phasing.py ' +\
                'script instead'
            nudie.log.error(s)
            return

        run(val['path'], 
                val['reference name'], 
                val['reference batch'], 
                val['experiment name'],
                val['experiment batch'], 
                plot=val['plot'], 
                force=val['force'])
    except Exception as e:
        nudie.log.exception(e)

if __name__ == '__main__':
    from sys import argv

    # turn on printing of errors
    level = nudie.logging.DEBUG
    nudie.show_errors(level)

    if len(argv) < 2:
        s = 'need a configuration file name as a parameter'
        nudie.log.error(s)
        sys.exit(-1)

    main(argv[1], level)
