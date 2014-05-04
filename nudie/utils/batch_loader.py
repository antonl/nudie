'''contains files to load the output of our LabView data taking stuff'''
import re
from pathlib import Path
import logging
log = logging.getLogger('nudie.batch_loader')
from datetime import date
import numpy as np

# This should really not be done here. I'll figure out where to move it later.
try:
    from .. import mount_point, data_folder
except ImportError as e:
    log.warning('could not load `data_folder` path.' +
            'Remember to set a default for these functions')
    data_folder = None

from .. import SpeFile

def load_job(job_name, when='today', batch_set=set([0]), data_path=data_folder):
    if data_path is None:
        raise RuntimeError('`data_path` needs to be set to a Path.')

    try:
        data_path = Path(data_path)
    except:
        s = 'could not convert {!r} to Path'.format(data_folder)
        log.error(s)
        raise ValueError(s)
    
    if when == 'today':
        # assume data_path goes to our data folder
        today = date.today().strftime('%y-%m-%d')
        log.info('using today\'s date ({:s}) in path'.format(today))
        data_path = data_path / Path(today)
    elif when is None:
        # assume data_path is a direct path to the directory containing batches
        log.info('`when` is None, assuming direct path to data')
    else:
        # assume when is a string to the right date folder in data_folder 
        log.info('`when` is \'{:s}\', assuming correct date')
        data_path = data_path / Path(when)

    if data_path.is_dir() is False:
        s = '`data_path` (`{!s}`) is not a valid directory.'.format(data_path)
        log.error(s) 
        raise ValueError(s)
    
    for n, batch in enumerate(batch_set):
        s = job_name + '-batch{:02d}'.format(batch)
        batch_path = data_path / Path(s)

        log.info('loading `{!s}`'.format(batch_path))

        if batch_path.is_dir() is False:
            log.warning('Attempted to load non-existing batch:')
            log.warning('`{!s}`'.format(batch_path))
            log.warning('skipped...')
            # Should it really skip on an error? 
            continue

        _examine_batch_dir(job_name, batch_path)

def _examine_batch_dir(job_name, path):
    '''file parses the directory structure and determines the loop settings 
    automagically

    internal function, path should be a Path object
    '''
    assert path.is_dir() == True

    # get t1 and t2 values available
    nt1, t1vals, nt2, t2vals = _examine_t1t2_files(path)

    job_pattern = re.compile(job_name + r'(\d+)-(\d+)-(\d+)\.spe')
    
    t1val_range = (0, 0)
    tableval_range = (0, 0)
    loop_range = (0, 0)

    # find the loop ranges
    for p in sorted(path.glob('*.spe')):
        log.debug('got file {!s} from batch dir'.format(p))
        myt1val, mytableval, myloopval = \
                (int(x) for x in job_pattern.match(p.name).groups())
        log.debug('t1val: {:d}\ttableval: {:d}\tloopval: {:d}'.format( \
                myt1val, mytableval, myloopval))

        t1val_range[0] = min(t1val_range[0], myt1val)
        t1val_range[1] = max(t1val_range[1], myt1val)

        tableval[0] = min(tableval[0], mytableval)
        tableval[1] = max(tableval[1], mytableval)

        loop_range[0] = min(loop_range[0], myloopval)
        loop_range[1] = max(loop_range[1], myloopval)

    log.debug('Got ranges:')
    log.debug('t1val: {!s}\ttableval: {!s}\tloopval: {!s}'.format( \
            t1val_range, tableval_range, loop_range))

def _examine_t1t2_files(path):
    assert path.is_dir() == True
    # look for t1pos and t2pos files

    try:
        with open(str(path / Path('t1pos.txt')), 'r') as f:
            t1vals = np.genfromtxt(f)
        with open(str(path / Path('t2pos.txt')), 'r') as f:
            t2vals = np.genfromtxt(f)
            t2vals.reshape((-1,2))

    except Exception as e:
        log.error('could not read t1pos and t2pos files')
        log.exception(e)
        raise RuntimeError('could not analyze t1pos or t2pos in' + \
            '{!s}'.format(path))
    
    return t1vals.shape[0], t1vals, t2vals.shape[0], t2vals
    
