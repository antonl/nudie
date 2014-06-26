'''
analyze stark TG signal using nudie
'''
from __future__ import division
import nudie
import numpy as np
import itertools as it
from datetime import date
import h5py
from matplotlib import pyplot

tg_name = 'd1d2-tg'
tg_batch = 0
when = '14-06-25'

# turn on printing of errors
nudie.show_errors(nudie.logging.INFO)

# create path to write our analysis to, if it doesn't exist already
analysis_folder = nudie.mount_point / '2D' / 'Analysis' / str(date.today())

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

# open hdf5 file to write to
file_path = analysis_folder / '{!s}-batch{:02d}.hdf5'.format(tg_name, tg_batch)
with h5py.File(str(file_path), 'w') as f:
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
    for t2 in it.islice(tg_info['t2_range'], 18, 19):
        t2_g = bg.create_group('{:02d}'.format(t2))
        t2_g.attrs['t2'] = t2

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
                    mdata_stark[k] = data[:, fo].mean(axis=-1) \
                        - data[:, fc].mean(axis=-1)

                    no = np.intersect1d(tags[0][0][k]['shutter open'], \
                            nostark_idx, assume_unique=True)
                    nc = np.intersect1d(tags[0][0][k]['shutter closed'], \
                            nostark_idx, assume_unique=True)
                    mdata_nostark[k] = data[:, no].mean(axis=-1) \
                        - data[:, nc].mean(axis=-1)
                
                for k, v in mdata_stark.items():
                    s_g[k] = v
                for k, v in mdata_nostark.items():
                    n_g[k] = v

                # store data as tg
                for s in loop_g.values():
                    s['time_tg'] = ((s['zero'][:] - s['none1'][:]) \
                            + (s['pipi'][:] - s['none2'][:]))   

    # this is how many pixels we have    
    npixels = data.shape[0]

    # Average tables and loops together, in case there were more than one
    for t2_g in bg.values(): 
        avg = np.zeros((data.shape[0],))
        t2_g.visititems(averager(avg, 'time_tg'))
        t2_g['mean_time_tg'] = avg
        pyplot.plot(avg)
    
    pyplot.show()
    # perform spectral interferometry
    wl = nudie.simple_wavelength_axis()[::-1]
    for t2_g in bg.values(): 
        tg = t2_g['mean_time_tg'][:]
        stuff = nudie.identify_prd_peak(wl, tg, all_info=True,
                axes=pyplot.axes())
        pyplot.show()

    '''
    # average together the individual pump probe spectra
    npixels = data.shape[0]
    pp_avg = np.zeros((npixels,))
    
    # visit written out file to find everything called 'pump_probe',
    # write that into pp_avg
    bg.visititems(averager(pp_avg, 'pump_probe'))
    try:
        del bg['mean_pump_probe']
    except:
        pass

    bg['mean_pump_probe'] = pp_avg
    
    ## Done with pump probe processing
    print("Finished pump probe processing")
    '''
