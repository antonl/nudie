import nudie
from matplotlib.pylab import *
import h5py
rcParams['figure.figsize'] = (14, 10)
from pathlib import Path

file_to_open = '15-03-23/M250V_TB_Asc_DOPA_TG_77K_100ps-batch00.h5'
save_folder = '15-05-07/figures'

nlevels = 50
levels_threshold = -50
dpi = 150

#probe_axis_limits = 0.330, 0.42 
probe_axis_limits = 0, 1

xsection_wavelength = 800

save_folder = Path(save_folder)

if not save_folder.exists():
    save_folder.mkdir(parents=True)
    
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 1.0.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.0 and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    #plt.register_cmap(cmap=newcmap)

    return newcmap

with h5py.File(file_to_open, 'r') as sf:
    tg = sf['raw transient-grating']
    
    wl3 = np.array(tg.dims[1][1]) # det wl
    idx =np.argmin(np.abs(wl3 - xsection_wavelength))
    f3 = np.array(tg.dims[1][0]) # det freq
    t2 = np.array(tg.dims[0][0]) # population time
    f3_slice = slice(np.argmin(np.abs(f3 - probe_axis_limits[0])),
            np.argmin(np.abs(f3 - probe_axis_limits[1])))    
    
    f3 = f3[f3_slice]
    
    res = np.abs(tg).T
    #res = np.rot90(np.abs(tg), -1)
    xsection = res[idx, :]
    res = res[f3_slice, :]
        
    vmax, vmin = ma.max(res), ma.min(res)
    ticker = mpl.ticker.MaxNLocator(nlevels)
    levels = ticker.tick_values(vmin, vmax)
    levels = levels[np.abs(levels) > levels_threshold] 
        
    midpoint = 1-vmax/(vmax+abs(vmin))
    cmap = shiftedColorMap(mpl.cm.RdBu_r, midpoint=midpoint)
        
    fig = figure()
    gs = GridSpec(2,2, height_ratios=[2,1], width_ratios=[5, 0.1]) 
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])

    ax1.contour(-t2, f3, res, 10, colors='k')
    cf = ax1.contourf(-t2, f3, res, levels=levels, cmap=cmap)
        
    ax1.set_ylabel(tg.dims[1].label)
    ax1.set_xlabel(tg.dims[0].label)            
        
    colorbar(cf, cax=ax3, use_gridspec=True)

    text_str = "Phased: {:s}\nnudie version: {:s}".format(
        'N/A',
        sf.attrs['nudie version'])                 
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax1.text(0.05, 0.95, text_str, transform=ax1.transAxes, 
        fontsize=14, verticalalignment='top', horizontalalignment='left', 
        bbox=props)

    ax2.plot(-t2, xsection)
    ax2.grid()
    gs.tight_layout(fig)
    show()
