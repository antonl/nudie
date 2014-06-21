'''
This example shows how to use nudie to do phasing of 2D data.
'''
from __future__ import division
import nudie
import numpy as np
import itertools as it
from datetime import date
import h5py
from matplotlib import pyplot

pp_name = 'r800-pp'
pp_batch = 1

spec_name = 'd1d2-2d'
spec_batch = 0 

when = '14-06-20'

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
file_path = analysis_folder / '{!s}-batch{:02d}.hdf5'.format(spec_name, spec_batch)
with h5py.File(str(file_path)) as f:
    # load up pp data to use
    pp_info = next(nudie.load_job(job_name=pp_name, batch_set=[pp_batch], 
        when=when))

    bg = f.require_group('pump_probe')

    # store some stuff for later in attributes
    bg.attrs['batch_name'] = pp_info['batch_name']
    bg.attrs['path'] = pp_info['batch_path']

    # load up all available data files
    for t2, table, loop in it.product(
            pp_info['t2_range'], 
            pp_info['table_range'], 
            pp_info['loop_range']):

        g = bg.require_group('{:02d}/{:02d}/{:02d}'.format(t2, table,loop))

        # first file requires special synchronization
        # this is the rule that determines that it is the first file
        first = 'first' if all([t2 == 0, table == 0, loop == 0]) else False
        
        # path to data
        current_path = nudie.data_folder / pp_info['when'] / pp_info['batch_name']

        # Load everything for the given t2, table, and loop value
        analogs = nudie.load_analogtxt(pp_info['job_name'], 
                current_path, t2, table, loop)
        cdata = nudie.load_camera_file(pp_info['job_name'], 
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
        
        # subtract the probe scatter by averaging the shutter closed data 
        # and subtracting it from each waveform of the shutter open data
        mdata = {}
        for k in phase_cycles:
            mdata[k] = data[:, tags[0][0][k]['shutter open']].mean(axis=-1) \
                    - data[:, tags[0][0][k]['shutter closed']].mean(axis=-1)
        
        # store each individual phase in the data file
        for k,v in mdata.items():
            try:
                del g[k]
            except:
                pass
            g[k] = v

        try:
            del g['pump_probe']
        except:
            pass

        # store data as pump probe
        g['pump_probe'] = ((mdata['zero'] - mdata['none1']) \
                + (mdata['pipi'] - mdata['none2']))   
    
    # average together the individual pump probe spectra
    npixels = data.shape[0]
    pp_avg = np.zeros((npixels,))
    
    # visit written out file to find everything called 'pump_probe',
    # write that into pp_avg
    bg.visititems(averager(pp_avg, 'pump_probe'))
    del bg['mean_pump_probe']
    bg['mean_pump_probe'] = pp_avg
    
    ## Done with pump probe processing
    print("Finished pump probe processing")

    wl = nudie.simple_wavelength_axis()[::-1]
    bg.visititems(plot_pp(pyplot.axes(),x=wl)) 
    pyplot.plot(wl, pp_avg, linewidth=1.5, color='k')
    pyplot.show()
