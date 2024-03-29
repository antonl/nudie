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
from math import floor
import sys

def make_objective(f, pp, tg, limits=(None, None)):
    '''create objective function for matching pump probe to TG'''
    axis = 2*np.pi*(f-f[0])[slice(*limits)]
    wtg = tg[slice(*limits)]
    wpp = pp[slice(*limits)]

    def fun(pval, info):
        curr = np.real(wtg*np.exp(-1j*np.polyval(pval, axis)))
        x, covx, infodict, msg, ier = leastsq(lambda y: y[0]*curr + y[1] - wpp, np.array([1, 0]), full_output=True)
        min_value = np.sum(infodict['fvec']**2)
        info.append((pval, x, min_value))
        return min_value

    return fun, axis

def accept_test(x_new, **kwargs):
    return True if np.abs(x_new[2] - np.pi/2) < np.pi else False

def make_nonlin_callback(f, pp, tg, info, prd_est=850.):
    import pyqtgraph as pg
    app = pg.mkQApp()

    mW = pg.GraphicsWindow(title="Optimization Progress")
    mW.resize(1000, 800)
    pOverlap = mW.addPlot(title='Pump probe vs TG fitting')
    pp_line = pOverlap.plot(f, pp, name='Pump probe', pen=pg.mkPen({'color': "FF0", 'width': 2}))
    tg_line = pOverlap.plot(f, np.real(tg), name='Projected TG', pen=pg.mkPen({'color': "0F0", 'width': 2}))
    pOverlap.setRange(yRange=(-1.1, 1.1), xRange=(f[0], f[-1]))
    mW.nextRow()
    pPhase = mW.addPlot(title='Phase')
    phase_line = pPhase.plot(f, np.unwrap(np.angle(tg)), name='Correction phase')
    pOverlap.disableAutoRange()
    axis = 2*np.pi*(f - f[0])
    #status_text = ax1.text(0.7, 1.10, "Not Running", transform=ax1.transAxes, fontsize=12, va='top')
    #param_text = ax1.text(0.7, 1.05, "Not Running", transform=ax1.transAxes, fontsize=12, va='top')

    def callback(pval, fval, accept):
        nonlin, lin, fval = info[-1]

        # evaluate function call
        phase = np.polyval(pval, axis)

        pTG = np.real(tg*np.exp(-1j*phase))
        scaledTG = lin[0]*pTG+lin[1]
        tg_line.setData(f, scaledTG)

        pval2 = np.array([pval[0], pval[1]-prd_est, pval[2]])
        phase2 = np.polyval(pval2, axis)
        phase_line.setData(f, phase2)

        app.processEvents()

    mW.show()
    return callback, mW, app

def plot_fit(f, correction_multiplier, correction_offset, pp_data, tg_data):
    import pyqtgraph as pg

    mW = pg.GraphicsWindow(title="Optimization Progress")
    mW.resize(1000, 800)

    plt = mW.addPlot(title="Phasing overlap")
    plt.plot(f, pp_data, pen=pg.mkPen({'color': "0F0", 'width': 2}))
    plt.plot(f, np.real(correction_multiplier*tg_data + correction_offset),
           pen=pg.mkPen({'color': "FF0", 'width': 2}))
    mW.show()
    return mW, plt

def plot_phased_2d(f1, f2, t2, dd):
    import pyqtgraph as pg
    mW = pg.GraphicsWindow(title="Optimization Progress")
    mW.resize(1000, 800)
    plt = mW.addPlot(title="Phasing overlap")
    # TODO: figure out how to plot 2D data

def run(path, pp_name, pp_batch, dd_name, dd_batch, plot=False, force=False,
        limits=None, phaselock_wl=650, central_wl=650, nsteps=300,
        niter_success=100, phasing_guess=[25, 850, 0],
        excitation_axis_zero_pad_to=-1, phase_time=10000, smooth_t1=False):

    path = Path(path)

    # create folder if it doesn't exist
    if not path.exists():
        s = 'Path {!s} doesn\'t exist. Please correct.'.format(path)
        nudie.log.error(s)
        raise FileNotFoundError(s)

    pp_file = path / '{:s}-batch{:02d}.h5'.format(pp_name, pp_batch)
    dd_file = path / '{:s}-batch{:02d}.h5'.format(dd_name, dd_batch)

    if not all([pp_file.exists(), dd_file.exists()]):
        s = 'Could not find one of the data files. ' +\
            'Please check that `{!s}` and `{!s}` exist.'.format(pp_file, dd_file)
        nudie.log.error(s)
        raise FileNotFoundError(s)

    # load 2D data
    with h5py.File(str(dd_file), 'r') as sf:
        # check that file is not already phased
        if 'phased' in sf.attrs:
            if not force:
                nudie.log.warning('Dataset already has a phasing correction. ' +\
                    'Use the `--force` flag to overwrite.')
                return

        if excitation_axis_zero_pad_to == -1:
            pad_to = sf.attrs['nt1']
        else:
            pad_to = excitation_axis_zero_pad_to

        # check that the axes I am creating have the right shape
        stored_pad_to = sf.attrs.get('excitation axis zero pad to', None)

        # this is pretty ugly
        if stored_pad_to is None:
            pass
        elif stored_pad_to != pad_to:
            s = 'phasing was attempted before but with a different zero pad ' +\
                'setting ({:d}). Please rerun raw 2D for this dataset.'.format(
                        stored_pad_to)
            nudie.log.error(s)
            raise RuntimeError(s)

        R = sf['raw rephasing']
        NR = sf['raw non-rephasing']

        dd_t2_idx = np.argmin(abs(np.array(R.dims[0][0]) - phase_time))
        dd_f = np.array(R.dims[2][0])
        dd_t1 = np.array(R.dims[1][0])
        prd_est = sf.attrs['probe lo delay estimate']

        R = np.array(R)
        NR = np.array(NR)

    # load pump-probe data
    with h5py.File(str(pp_file), 'r') as sf:
        pp = sf['averaged pump-probe']
        pp_t2idx = np.argmin(abs(np.array(pp.dims[0][0]) - phase_time))
        pp_data = np.array(pp[pp_t2idx])
        pp_f = np.array(pp.dims[1][0])

    assert np.allclose(pp_f, dd_f), "detection axes do not match!"

    t1_zero_idx = np.argmin(abs(dd_t1 - 0))
    tg_data = 0.5*(R[dd_t2_idx, t1_zero_idx] + NR[dd_t2_idx, t1_zero_idx])

    info = deque([])
    test, axis  = make_objective(dd_f, pp_data, tg_data, limits=limits)

    if plot:
        callback, mW, app = make_nonlin_callback(dd_f, pp_data, tg_data, info, prd_est)
    else:
        callback = None

    minimizer_kwargs={'method': 'L-BFGS-B', 'args' : (info,), 'bounds' : [(None, None), (None, None), (0, np.pi)]}
    res = basinhopping(test, phasing_guess, niter=nsteps, minimizer_kwargs=minimizer_kwargs, stepsize=50,
        niter_success=niter_success, disp=True, accept_test=accept_test, callback=callback)

    # pull out best parameters
    while len(info) > 0:
        nonlin, lin, min_value = info.pop()
        if min_value == res.fun:
            del info
            break

    global_phase = np.polyval(nonlin, 2*np.pi*(dd_f-dd_f[0]))
    correction_multiplier = lin[0]*np.exp(-1j*np.polyval(nonlin, 2*np.pi*(dd_f-dd_f[0])))
    correction_offset = lin[1]

    if plot:
        mW = plot_fit(dd_f, correction_multiplier, correction_offset, pp_data, tg_data)

    # open 2d file and write out corrected phase
    with h5py.File(str(dd_file), 'a') as sf:
        gp = sf.require_dataset('global phase', shape=global_phase.shape,
                dtype=float)
        gp[...] = global_phase
        cm = sf.require_dataset('correction multiplier', shape=(dd_f.shape[0],),
                dtype=complex)
        cm[...] = correction_multiplier
        co = sf.require_dataset('correction offset', shape=(), dtype=float)
        co[...] = correction_offset
        phasingpp = sf.require_dataset('phasing pump probe',
                                       shape=pp_data.shape, dtype=float)
        phasingpp[...] = pp_data
        ptg = sf.require_dataset('phased tg', shape=tg_data.shape, dtype=float)
        ptg[...] = np.real(tg_data*correction_multiplier + correction_offset)

    # fft R and NR, deleting temporaries so that we don't store too much stuff
    # in memory

    with h5py.File(str(dd_file), 'a') as sf:
        # create excitation axis
        t1 = sf['axes/t1']

        dt1 = abs(t1[1] - t1[0])
        f1 = np.fft.fftshift(np.fft.fftfreq(pad_to, dt1))

        f1_ax = sf['axes'].require_dataset('raw excitation frequency',
            shape=f1.shape, dtype=float)
        f1_ax[...] = f1

        # trying frank's method of phase-locking axis
        # this is due to a sign error in the Dazzler code
        # Note: This is now fixed. Bug was due to one extra fftshift
        C = nudie.spectrometer.speed_of_light
        f1_pl = f1 + C/phaselock_wl

        f1_ax_pl = sf['axes'].require_dataset('phase-locked excitation frequency',
            shape=f1.shape, dtype=float)
        f1_ax_pl[...] = f1_pl

        spectra_shape = (sf.attrs['nt2'], f1.shape[0], dd_f.shape[0])

        # Window the raw data to avoid wings
        t1_len = t1.shape[0]
        if smooth_t1:
            window_sym = get_window(('kaiser', 8.0), 2*t1_len, fftbins=True)
            window = window_sym[t1_len:]
        else:
            window = np.ones((t1_len,))

        window_func = lambda x: np.einsum('ijk,j->ijk', x, window)
        # average first and last point together
        def apply_boundary_conds(S, axis=1):
            # get the first and last indices
            ind_start = tuple([0 if i == axis else slice(None) \
                         for i in range(S.ndim)])
            ind_end = tuple([-1 if i == axis else slice(None) \
                               for i in range(S.ndim)])
            avg = 0.5*(S[ind_start] + S[ind_end])
            print(ind_start)

            # replace original signal at boundaries with average
            S[ind_start]  = avg

        # zero pad data and flip axes
        apply_boundary_conds(R, axis=1)
        Rw1 = np.fft.fftshift(np.fft.fft(window_func(R), axis=1, n=pad_to), axes=1)
        phased_Rw1 = correction_multiplier*Rw1 + correction_offset
        r = sf.require_dataset('phased rephasing', shape=spectra_shape,
                dtype=complex)
        r[...] = phased_Rw1
        del R, phased_Rw1

        apply_boundary_conds(NR, axis=1)
        NRw1 = np.fft.fftshift(np.fft.fft(window_func(NR), axis=1, n=pad_to), axes=1)
        phased_NRw1 = correction_multiplier*NRw1 + correction_offset
        nr = sf.require_dataset('phased non-rephasing', shape=spectra_shape,
                dtype=complex)
        nr[...] = phased_NRw1
        del NR, phased_NRw1

        # properly combine the rephasing and nonrephasing
        roll_by = 1 if pad_to % 2 == 0 else 0

        Sw1 = 0.5*(np.roll(Rw1[:, ::-1], shift=roll_by, axis=1) + NRw1)
        # note: Sw1 has essentially already been fftshift-ed because Rw1 and NRw1 already are

        phased_2D = correction_multiplier*Sw1 + correction_offset
        dd = sf.require_dataset('phased 2D', shape=spectra_shape,
                dtype=complex)
        dd[...] = phased_2D

        sf.attrs['nonlinear coefficients'] = nonlin
        sf.attrs['linear coefficients'] = lin
        sf.attrs['fitness'] = min_value

        sf.attrs['phase lock wavelength'] = phaselock_wl
        sf.attrs['central wavelength'] = central_wl
        sf.attrs['excitation axis zero pad to'] = pad_to

        # attach dimension scales
        r.dims.create_scale(f1_ax, 'raw excitation frequency')
        r.dims.create_scale(f1_ax_pl, 'phase-locked excitation frequency')

        for x in [r, nr, dd]:
            x.dims[0].label = 'population time'
            x.dims[0].attach_scale(sf['axes/t2'])
            x.dims[1].label = 'pump frequency'
            x.dims[1].attach_scale(f1_ax)
            x.dims[1].attach_scale(f1_ax_pl)
            x.dims[2].label = 'detection frequency'
            x.dims[2].attach_scale(sf['axes/detection frequency'])

        # attach phased 2D attribute
        sf.attrs['phased'] = True
        sf.attrs['phasing timestamp'] = arrow.now().format('DD-MM-YYYY HH:mm')
        sf.attrs['nudie version'] = nudie.version

        del phased_2D

def main(config, verbosity=nudie.logging.INFO):
    nudie.show_errors(verbosity)

    try:
        try:
            val = nudie.parse_config(config, which='phasing')['phasing']
        except ValueError as e:
            nudie.log.error('could not validate file. Please check ' +\
                'configuration options.')
            return

        if val['copy']:
            s = '`copy` flag is set. You should be using the apply-phase.py ' +\
                'script instead'
            nudie.log.error(s)
            return

        run(val['path'],
            val['reference name'],
            val['reference batch'],
            val['experiment name'],
            val['experiment batch'],
            plot=val['plot'],
            force=val['force'],
            limits=val['pixel range to fit'],
            phaselock_wl=val['phaselock wl'],
            central_wl=val['central wl'],
            nsteps=val['nsteps'],
            niter_success=val['nstep success'],
            phasing_guess=val['phasing guess'],
            phase_time=val['phasing t2'],
            excitation_axis_zero_pad_to=val['excitation axis zero pad to'],
            smooth_t1=val['smooth t1'])
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
