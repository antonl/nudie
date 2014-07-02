'''
analyze stark TG signal using nudie
'''
from __future__ import division
import nudie
import numpy as np
import itertools as it
from datetime import date
import h5py
import pdb
from matplotlib import pyplot
from scipy.signal import argrelmax
from pathlib import Path

tg_name = 'd1d2-tg'
tg_batch = 1
when = '14-06-28'

# turn on printing of errors
nudie.show_errors(nudie.logging.INFO)

# create path to write our analysis to, if it doesn't exist already
analysis_folder = Path('.')
#analysis_folder = nudie.mount_point / '2D' / 'Analysis' / str(date.today())

try:
    analysis_folder.mkdir(parents=True)
except OSError as e:
    pass

# define averager function for visiting hdf5 arrays
def averager(outp, dataset_name):
    N = 0.
    def wrapped(name, obj):
        nonlocal outp, dataset_name, N
        if dataset_name in name:
            outp[:] = (outp*N + obj[:])/(N+1)
            N += 1
    return wrapped

def plot_pp(ax, x=None):
    if x is None:
        wl = nudie.simple_wavelength_axis()
    else: 
        wl = x

    def wrapped(name, obj):
        nonlocal ax, wl
        if 'pump_probe' in name:
            ax.plot(wl, obj)
        
    return wrapped

def LO_selector(prd_found, wl):
    import re
    matcher = re.compile(r'^[\w/]*(stark|no stark)')

    def wrapped(name, obj):
        nonlocal prd_found, wl, matcher
        if (type(obj) == h5py.Group) and matcher.match(name):
            print('starting ' + str(obj.name))
            # got a group containing phases
            num_shots = []
            for phase in obj.values():
                num_shots.append(phase.shape[1])
            length = min(num_shots)
            
            tmp = {}
            for phase in ['zero', 'none1', 'pipi', 'none2']:
                print('\tstarting ' + str(phase))
                # run spectral interferometery
                tmp[phase] = np.zeros((1340, length), dtype=complex)
                for i in range(length):
                    print(i)
                    _, _, E_sig, E_lo = nudie.select_prd_peak(wl,
                            obj[phase][:,i], known_prd=prd_found) 
                    tmp[phase][:, i] = E_sig/E_lo
            res = (tmp['zero'] - tmp['none1']) \
                    + (tmp['pipi'] - tmp['none2'])
            obj['mean_tg'] = res.mean(axis=-1)
    return wrapped

def find_mean_tg(name):
    if 'mean_tg' in name:
        return name

# do spectrometer calibration
calib_spec_file = nudie.SpeFile(nudie.data_folder / '14-06-28' \
        / 'Calib_Spec_650nm_600g.SPE')

data = np.squeeze(calib_spec_file.data).mean(axis=-1)

fig, (ax1, ax2) = pyplot.subplots(2, 1)

# baseline subtraction
px = np.arange(data.shape[0])
pv = np.polyfit(px, data, 2)
data -= np.polyval(pv, px)

# peak fitting
res = argrelmax(data, order=10)[0]
select = data[res] > 0.01*np.max(data)

peaks = np.array([706.722, 696.543, 668.2968, 626.3688, 625.1348, 604.3008, 593.4566])

color = it.cycle(['g','c'])
lines_list = [(i*nudie.spectrometer.hg_ar_lines, next(color)) for i in range(1, 3)]

corr = np.polyfit(res[select], peaks, 2)
wl = np.polyval(corr, px)

for lines, c in lines_list:
    ax1.vlines(lines, -0.3*np.max(data), 0, c, linewidth=2)
ax1.plot(wl, data)
ax1.vlines(np.polyval(corr, res[select]), 0, 0.3*np.max(data), 'r')
ax1.set_ylabel('Intensity / AU')
ax1.set_xlabel('Wavelength / nm')
ax1.set_xlim(min(wl), max(wl))
ax1.text(660, 5000, r'$y = {:.3g} \cdot x^2 + {:.3g} \cdot x + {:.3g}$'.format(*corr), fontsize=16)
ax2.plot(px, np.polyval(corr, px))
ax2.set_ylabel('Wavelength / nm')
ax2.set_xlabel('Pixel');
pyplot.show()

# open hdf5 file to write to
file_path = analysis_folder / '{!s}-batch{:02d}.hdf5'.format(tg_name, tg_batch)
with h5py.File(str(file_path), 'w') as f:
    # save spectrometer calibration
    spec = f.require_group('spectrometer calibration')
    spec['spectrum'] = data
    spec['pixel map'] = corr
    spec['axis'] = wl

    # load up pp data to use
    tg_info = next(nudie.load_job(job_name=tg_name, batch_set=[tg_batch], 
        when=when))

    bg = f.require_group('tg')

    # store some stuff for later in attributes
    bg.attrs['batch_name'] = tg_info['batch_name']
    bg.attrs['path'] = tg_info['batch_path']

    # path to data
    current_path = nudie.data_folder / tg_info['when'] / tg_info['batch_name']

    # load up all available data files
    for t2 in it.islice(tg_info['t2_range'], 75, 76):
        t2_g = bg.create_group('{:02d}'.format(t2))
        t2_g.attrs['t2'] = tg_info['t2'][t2]

        for table in tg_info['table_range']:
            table_g = t2_g.create_group('{:02d}'.format(table))
            table_g.attrs['table'] = table

            for loop in tg_info['loop_range']:
                loop_g = table_g.create_group('{:02d}'.format(loop))
                loop_g.attrs['loop'] = loop

                # first file requires special synchronization
                # this is the rule that determines that it is the first file
                first = 'first' if all([t2 == 0, table == 0, loop == 0]) else False

                # Load everything for the given t2, table, and loop value
                analogs = nudie.load_analogtxt(tg_info['job_name'], 
                        current_path, t2, table, loop)
                cdata = nudie.load_camera_file(tg_info['job_name'], 
                        current_path, t2, table, loop, force_uint16=True)

                # Synchronize it, trimming some frames in the beginning and end
                data, (a1, a2) = nudie.trim_all(*nudie.synchronize_daq_to_camera(cdata, 
                    analog_channels=analogs, which_file=first))
                start_idxs, period = nudie.detect_table_start(a1)

                # determine where the shutter is
                shutter_open, shutter_closed, duty_cycle = \
                        nudie.determine_shutter_shots(data)

                # tag the phases
                shutter_info = {'last open idx': shutter_open, 
                        'first closed idx': shutter_closed}
                phase_cycles = ['none1', 'zero', 'none2', 'pipi']
                tags = nudie.tag_phases(start_idxs, period, tags=phase_cycles, 
                        nframes=data.shape[1], shutter_info=shutter_info)
                
                # determine stark on shots
                stark_idx, nostark_idx = np.where(a2 > 0.2)[0], \
                        np.where(a2 <= 0.2)[0]
                
                # subtract the probe scatter by averaging the shutter closed data 
                # and subtracting it from each waveform of the shutter open data
                mdata_stark = {}
                mdata_nostark = {}
                s_g, n_g = loop_g.create_group('stark'), \
                        loop_g.create_group('no stark') 

                for k in phase_cycles:
                    fo = np.intersect1d(tags[0][0][k]['shutter open'], stark_idx,
                            assume_unique=True)
                    fc = np.intersect1d(tags[0][0][k]['shutter closed'], \
                            stark_idx, assume_unique=True)
                    mdata_stark[k] = data[:, fo] \
                            - data[:, fc].mean(axis=-1)[:, np.newaxis]

                    no = np.intersect1d(tags[0][0][k]['shutter open'], \
                            nostark_idx, assume_unique=True)
                    nc = np.intersect1d(tags[0][0][k]['shutter closed'], \
                            nostark_idx, assume_unique=True)
                    mdata_nostark[k] = data[:, no] \
                            - data[:, nc].mean(axis=-1)[:, np.newaxis]
                
                for k, v in mdata_stark.items():
                    s_g[k] = v
                for k, v in mdata_nostark.items():
                    n_g[k] = v

    # this is how many pixels we have    
    npixels = data.shape[0]
    wl = f['spectrometer calibration/axis'][:]

    if bg.get('probe_ref_delay') is None:
        # pick a T2 after time zero to determine the PRD
        t2 = tg_info['nt2']//2 
        prd_estimate_spectrum = bg['{:02d}/00/00/no stark/pipi'.format(t2)]\
                [:].mean(axis=-1)
        prd_guess, freq, E_sig, E_lo = nudie.select_prd_peak(wl, 
                prd_estimate_spectrum, axes=pyplot.axes())
        pyplot.show()

    #bg.visititems(LO_selector(prd_guess, wl))
    
    '''
    tg_scan = bg['spectrum'] = np.zeros((1340, tg_info['nt2']), dtype=complex)
    for i, t2_item in enumerate(bg.values()):
        if type(t2_item) == h5py.Group:
            t2 = t2_item.attrs['t2']
            tg = t2_item.visit(find_mean_tg)
            tg_scan[:, i] = t2_item[tg][:]

    pyplot.plot(freq, np.abs(tg_scan))
    pyplot.show()
    '''
    #wl, E_sig = nudie.freq_to_wavelen(freq, E_sig)
    #_, E_lo = nudie.freq_to_wavelen(freq, E_lo)

    #pyplot.plot(wl, np.real(E_sig)/np.real(E_lo))
    #pyplot.plot(wl, np.real(E_lo))
    #pyplot.show()

