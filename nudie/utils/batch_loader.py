'''contains files to load the output of our LabView data taking stuff'''

import logging
log = logging.getLogger('nudie.batch_loader')

from pathlib import Path

# This should really not be done here. I'll figure out where to move it later.
try:
    from .. import mount_point
    data_folder = mount_point / Path('2D/Data')
except ImportError as e:
    log.warning('could not load `mount_point` path.' +
            'Remember to set a default for these functions')
    data_folder = None

from .. import SpeFile

def load_job(job_name, batch_set=set([0]), data_path=data_folder):
    if data_path is None:
        raise RuntimeError('`data_path` needs to be set to a Path.')

    try:
        data_path = Path(data_path)
    except:
        s = 'could not convert {!r} to Path'.format(data_folder)
        log.error(s)
        raise ValueError(s)


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

        yield (n, batch)

def _examine_batch_dir(path):
    '''file parses the directory structure and determines the loop settings 
    automagically

    internal function, path should be a Path object
    '''
    assert path.is_dir() == True
    
    raise NotImplementedError('still working on this')

def _examine_t1t2_files(path):
    assert path.is_dir() == True

    

