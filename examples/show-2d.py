from matplotlib.pylab import *
import h5py
rcParams['figure.figsize'] = (16, 8)
import numpy.ma as ma

file_to_open = '15-06-05/Bchla-2d-batch01.h5'
nlevels = 50
levels_threshold = 0.5

pump_axis_limits = 0.335, 0.435
probe_axis_limits = 0.335, 0.435

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
    dd = sf['phased 2D']
    f1 = np.array(dd.dims[1][1]) # excitation freq
    f3 = np.array(dd.dims[2][0]) # det freq
    t2 = np.array(dd.dims[0][0]) # population time
    
    f1_slice = slice(np.argmin(np.abs(f1 - pump_axis_limits[0])),
            np.argmin(np.abs(f1 - pump_axis_limits[1])))
    f3_slice = slice(np.argmin(np.abs(f3 - probe_axis_limits[0])),
            np.argmin(np.abs(f3 - probe_axis_limits[1])))
    
    res = np.rot90(np.real(dd[0]), -1)
    f1 = f1[f1_slice]
    f3 = f3[f3_slice]
    res = res[f3_slice, f1_slice]
    
    #res = ma.masked_where(abs(res) < 0.25, res, copy=False)
    vmax, vmin = ma.max(res), ma.min(res)
    ticker = mpl.ticker.MaxNLocator(nlevels)
    levels = ticker.tick_values(vmin, vmax)
    levels = levels[np.abs(levels) > levels_threshold] 
    
    midpoint = 1-vmax/(vmax+abs(vmin))
    cmap = shiftedColorMap(mpl.cm.RdBu_r, midpoint=midpoint)
    
    ca = contour(f1, f3, res, levels=levels, colors='k')
    cb = contourf(f1, f3, res, levels=levels, cmap=cmap)
    
    ylabel(dd.dims[2].label)
    xlabel(dd.dims[1].label)    
    
    gca().add_line(Line2D([0, 1], [0, 1], linewidth=1.5, color='k'))
    #plot([0.33, 0.42], [0.33, 0.42], linewidth=3, linestyle='-', color='b')    

    colorbar()
    text_str = "Phased: {:s}\nnudie version: {:s}".format(
        sf.attrs['phasing timestamp'],
        sf.attrs['nudie version'])                 
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    text(0.99, 0.01, text_str, transform=gca().transAxes, fontsize=14, verticalalignment='bottom', horizontalalignment='right', bbox=props)
    text(0.05, 0.95, "T = {:3.0f}fs".format(np.abs(t2[0])), transform=gca().transAxes, fontsize=20, verticalalignment='top')
    gca().set_aspect('equal', 'datalim')

show()
