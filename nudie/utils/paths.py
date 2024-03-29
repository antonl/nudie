# make sure things work in a uniform way
from __future__ import division, absolute_import, unicode_literals, print_function
import logging
import pdb

__all__ = ['gather_information', 'setup_paths']

log = logging.getLogger('nudie.paths')

def gather_information():
    ''' collect information about running machine
    '''
    import platform
    info = {}
    info['arch'] = platform.architecture()[0]
    info['system'] = platform.system()
    info['py'] = platform.python_version()
    info['pytuple'] = platform.python_version_tuple()
    return info

def setup_paths():
    ''' find the location of the data drive for the ogilvie lab.

    This function creates a global variable `mount_point` that gives the
    location of the data drive, assuming that it is in conventional locations.
    If that path doesn't work, it raises a `RuntimeError`. Requires pathlib.
    '''
    info = gather_information()
    
    try:
        from pathlib import Path
    except ImportError as e:
        log.error('could not find `pathlib` module. please run python 3.4 or install pathlib')
        raise e
    
    sys = info['system']
    if sys == 'Darwin':
        # running on OS X
        mount_point = Path('/Volumes/jogilvie')
    elif sys == 'Linux':
        mount_point = Path('/mnt/data')
    elif sys == 'Windows':
        # sometimes it is P, other times it is Z
        if Path('P:/').exists():
            mount_point = Path('P:/')
        elif Path('Z:/').exists():
            mount_point = Path('Z:/')
        else:
            mount_point = None

    # check that the mount point exists
    if mount_point is None or not mount_point.exists():
        s = 'mount_point path is incorrect. `{mp!s}` does not exist.'.format(mp=mount_point)
        log.error(s)
        raise RuntimeError(s)

    data_folder = mount_point / Path('2D/Data')

    # check that the mount point exists
    if not data_folder.exists():
        s = 'data_folder does not exist `{df!s}`' + \
            'does not exist.'.format(df=data_folder)
        log.error(s)
        raise RuntimeError(s)
    
    log.info('mount_point variable is set to `{mp!s}`'.format(mp=mount_point))
    log.info('data_folder variable is set to `{df!s}`'.format(df=data_folder))

    return mount_point, data_folder

