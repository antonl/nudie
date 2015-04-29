"""
Definition of configuration file format used by the analysis scripts, and some
utility functions for cleaning up the config file
"""

from functools import lru_cache
import configparser 
from pathlib import Path
from collections import OrderedDict
import logging
log = logging.getLogger('nudie.schema')

import voluptuous
from voluptuous import Schema, Required, Coerce, Optional, REMOVE_EXTRA

class BoolStr(str):
    truthy = ['true', 'yes', '1', 'on']
    falsy = ['false', 'no', '0', 'off']
    
    def __bool__(self):
        s = self.lower()
        if s in self.truthy:
            return True
        elif s in self.falsy:
            return False
        else:
            raise ValueError('could not coerce `{:s}` to bool'.format(self))

    def __eq__(self, other):
        return bool(self) == other

    def __repr__(self):
        return 'BoolStr(' + self + ')'

class IntList(list):
    def __init__(self, s):
        if s == '' or s is None:
            super(IntList, self).__init__([])
        else:            
            super(IntList, self).__init__(map(int, s.split(',')))

class FloatList(list):
    def __init__(self, s):
        if s == '' or s is None:
            super(FloatList, self).__init__([])
        else:            
            super(FloatList, self).__init__(map(float, s.split(',')))

def as_dict(config):
    """
    Converts a ConfigParser object into a dictionary.

    The resulting dictionary has sections as keys which point to a dict of the
    sections options as key => value pairs.

    Taken from http://stackoverflow.com/a/23944270
    """
    the_dict = OrderedDict({})
    for section in config.sections():
        the_dict[section] = {}
        for key, val in config.items(section):
            the_dict[section][key] = val
    return the_dict

# Optional and Required are a little interchangable because I pass default
# parameters anyway
schema = Schema({
  'pump probe': {
        Required('jobname'): str,
        Required('batch'): Coerce(int),
        Required('when'): str, # probably should validate to a valid date
        Required('wavelengths'): str,
        Required('analysis path'): str,
        Optional('plot'): Coerce(BoolStr),
        Optional('exclude'): Coerce(IntList),
        },
  '2d': {
        Required('jobname'): str,
        Required('batch'): Coerce(int),
        Required('when'): str, # probably should validate to a valid date
        Required('wavelengths'): str,
        Required('analysis path'): str,
        Optional('waveforms per table'): Coerce(int),
        Optional('central wl'): Coerce(float),
        Optional('phaselock wl'): Coerce(float),
        Optional('plot'): Coerce(BoolStr),
        Optional('force'): Coerce(BoolStr),
        Optional('stark'): Coerce(BoolStr),
        Optional('field on theshold'): Coerce(float),
        Optional('probe ref delay'): Coerce(float), 
        Optional('lo width'): Coerce(float),
        Optional('dc width'): Coerce(float),
        Optional('zero pad to'): Coerce(int),
        Optional('gaussian power'): Coerce(float),
        Optional('pump chop'): Coerce(BoolStr),
        },
    'phasing': {
        Required('path'): str,
        Required('reference name'): str,
        Required('reference batch'): Coerce(int),
        Required('2d name'): str,
        Required('2d batch'): Coerce(int),
        Optional('copy'): Coerce(BoolStr),
        Optional('phasing guess'): Coerce(FloatList),
        Optional('zero pad to'): Coerce(int),
        Optional('nsteps'): Coerce(int),
        Optional('nstep success'): Coerce(int),
        Optional('pixel range to fit'): Coerce(IntList), # check that lb < ub
        Optional('force'): Coerce(BoolStr),
        Optional('plot'): Coerce(BoolStr),
        Optional('phaselock wl'): Coerce(float),
        Optional('central wl'): Coerce(float),
        Optional('phasing t2'): Coerce(float),
        },
    }, extra=REMOVE_EXTRA)

defaults = {
    'pump probe': {
        'exclude' : '',
        'plot'    : False,
        },
    '2d' : {
        'plot' : False,
        'force' : False,
        'pump chop' : False,
        'stark' : False,
        'field on theshold': 0.2,
        'waveforms per table' : 40,
        'central wl' : 650,
        'phaselock wl'  : 650,
        'probe ref delay' : 850.,
        'lo width' : 200,
        'dc width' : 200,
        'zero pad to' : 2048,
        'analysis path' : 'analyzed',
        'gaussian power' : 2,
        },
    'phasing': {
        'force': False,
        'plot' : False,
        'copy': False,
        'phasing guess': [25, 850, 0],
        'zero pad to' : 2048,
        'nsteps' : 300,
        'nstep success' : 100,
        'pixel range to fit': (0, 1340),
        'zero pad to' : 128,
        'phaselock wl' : 650,
        'central wl' : 650,
        'phasing t2' : 10000,
        },
    }

def parse_config(path):
    path = Path(path)
    log.debug('parse_config got `{!s}` as input file'.format(path))
    
    if not path.is_file():
        s = '`{!s}` does not exist!'.format(path)
        log.error(s)
        raise ValueError(s)

    try:
        cfg = configparser.ConfigParser(default_section='common',
                interpolation=configparser.ExtendedInterpolation(), 
                empty_lines_in_values=False, 
                allow_no_value=True)
        log.debug('setting defaults:')
        for sec, val in defaults.items():
            log.debug('  {!s}: {!s}'.format(sec, val))
        cfg.read_dict(defaults)
        cfg.read(str(path))

        log.debug('parsed:')
        for sec, val in cfg.items():
            log.debug('  {!s}:'.format(sec))
            for subsec, subval in val.items():
                log.debug('    {!s}: {!s}'.format(subsec, subval))

        log.debug('validating to schema')
        validated = schema(as_dict(cfg))

        log.debug('final schema is:')
        for sec, val in validated.items():
            log.debug('  {!s}:'.format(sec))
            for subsec, subval in val.items():
                log.debug('    {!s}: {!s}'.format(subsec, subval))
    except configparser.Error as e:
        s = 'error while parsing file `{!s}`'.format(path)
        log.error(s)
        log.error(e)
        raise RuntimeError(s)
    except voluptuous.MultipleInvalid as e:
        log.error(e)
        raise RuntimeError(s)

    return validated

