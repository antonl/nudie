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
import matplotlib.pyplot as mpl

def run(path, ref_name, ref_batch, dd_name, dd_batch, plot=False,
        force=False, central_wl=-1, phaselock_wl=-1, stark=False):
    path = Path(path)

    # create folder if it doesn't exist
    if not path.exists():
        s = 'Path {!s} doesn\'t exist. Please correct.'.format(path)
        nudie.log.error(s)
        raise FileNotFoundError(s)

    ref_file = path / '{:s}-batch{:02d}.h5'.format(ref_name, ref_batch)
    dd_file = path / '{:s}-batch{:02d}.h5'.format(dd_name, dd_batch)
    
    if not all([ref_file.exists(), dd_file.exists()]):
        s = 'Could not find one of the data files. ' +\
            'Please check that `{!s}` and `{!s}` exist.'.format(ref_file, dd_file)
        nudie.log.error(s)
        raise FileNotFoundError(s)

    # load phased data
    with h5py.File(str(ref_file), 'r') as sf:
        if sf.attrs['phased'] != True:
            s = 'phasing dataset does not contain a phasing correction. ' +\
                'Cannot reuse it for phasing!'
            nudie.log.error(s)
            raise RuntimeError(s)

        excitation_axis_pad_to = sf.attrs['excitation axis pad to']
        correction_multiplier = np.array(sf['correction multiplier'])
        correction_offset = np.array(sf['correction offset'])
        
    # load 2D data
    with h5py.File(str(dd_file), 'a') as sf:
        # check that file is not already phased
        if 'phased' in sf.attrs:
            if not force:
                nudie.log.warning('Dataset already has a phasing correction. ' +\
                    'Use the force flag to overwrite.')
                return

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
        C = nudie.spectrometer.speed_of_light
        f1_pl = f1 + C/(2*central_wl - phaselock_wl)

        f1_ax_pl = sf['axes'].require_dataset('phase-locked excitation frequency',
            shape=f1.shape, dtype=float)
        f1_ax_pl[:] = f1_pl 

        if stark:
            spectra_shape = (sf.attrs['nt2'], sf.attrs['nstark'], f1.shape[0], dd_f.shape[0])
        else:
            spectra_shape = (sf.attrs['nt2'], f1.shape[0], dd_f.shape[0])

        # zero pad data and flip axes
        if stark:
            Rw1 = np.fft.fft(R, axis=2, n=excitation_axis_pad_to)
            NRw1 = np.fft.fft(NR, axis=2, n=excitation_axis_pad_to)
            Sw1 = np.fft.fftshift(0.5*(Rw1[:, :, ::-1] + NRw1), axes=(2,))
        else:
            Rw1 = np.fft.fft(R, axis=1, n=excitation_axis_pad_to)
            NRw1 = np.fft.fft(NR, axis=1, n=excitation_axis_pad_to)
            Sw1 = np.fft.fftshift(0.5*(Rw1[:, ::-1] + NRw1), axes=(1,))

        phased_Rw1 = correction_multiplier*Rw1 + correction_offset
        r = sf.require_dataset('phased rephasing', shape=spectra_shape,
                dtype=complex, data=phased_Rw1) 

        phased_NRw1 = correction_multiplier*NRw1 + correction_offset
        nr = sf.require_dataset('phased non-rephasing', shape=spectra_shape,
                dtype=complex, data=phased_NRw1) 

        phased_2D = correction_multiplier*Sw1 + correction_offset
        dd = sf.require_dataset('phased 2D', shape=spectra_shape, 
                dtype=complex, data=phased_2D)

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

        # attach phased 2D attribute
        sf.attrs['phased'] = True
        sf.attrs['phasing timestamp'] = arrow.now().format('DD-MM-YYYY HH:MM')
        sf.attrs['nudie version'] = nudie.version

        if plot:
            if stark:
                mpl.figure()
                mpl.contourf(f1_pl, sf['axes/detection frequency'], 
                    np.rot90(np.real(phased_2D[0][0] - phased_2D[0][1]), -1), 50)
                mpl.colorbar()
                mpl.figure()
                mpl.contourf(f1_pl, sf['axes/detection frequency'], 
                    np.rot90(np.real(phased_2D[0][1]), -1), 50)
                mpl.colorbar()
                mpl.show()
            else:
                mpl.contourf(f1_pl, sf['axes/detection frequency'], 
                    np.rot90(np.real(phased_2D[0]), -1), 50)
                mpl.colorbar()
                mpl.show()

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
            val = nudie.parse_config(argv[1], which='phasing')['phasing']
        except ValueError as e:
            nudie.log.error('could not validate file. Please check ' +\
                'configuration options.')
            sys.exit(-1)

        if val['copy'] is False:
            s = '`copy` flag is not set. You should be using the phasing.py ' +\
                'script instead'
            nudie.log.error(s)
            raise RuntimeError(s)

        run(val['path'], 
                val['reference name'], 
                val['reference batch'], 
                val['2d name'],
                val['2d batch'], 
                plot=val['plot'], 
                force=val['force'],
                phaselock_wl=val['phaselock wl'], 
                central_wl=val['central wl'],
                stark=val['stark'])
    except Exception as e:
        raise e
