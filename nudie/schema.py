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
        s = self.strip().lower()
        if s in self.truthy:
            return True
        elif s in self.falsy:
            return False
        else:
            raise ValueError('could not coerce `{:s}` to bool'.format(self))

    def __eq__(self, other):
        return bool(self) == other

    def __ne__(self, other):
        return not (bool(self) == other)

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
schema_pp = Schema({
    Required('jobname'): str,
    Required('batch'): Coerce(int),
    Required('when'): str, # probably should validate to a valid date
    Required('wavelengths'): str,
    Required('analysis path'): str,
    Required('data path'): str,
    Optional('plot'): Coerce(BoolStr),
    Optional('exclude'): Coerce(IntList),
    }, extra=REMOVE_EXTRA)

schema_2d = Schema({
    Required('jobname'): str,
    Required('batch'): Coerce(int),
    Required('when'): str, # probably should validate to a valid date
    Required('wavelengths'): str,
    Required('analysis path'): str,
    Required('data path'): str,
    Optional('waveforms per table'): Coerce(int),
    Optional('central wl'): Coerce(float),
    Optional('phaselock wl'): Coerce(float),
    Optional('plot'): Coerce(BoolStr),
    Optional('force'): Coerce(BoolStr),
    Optional('stark'): Coerce(BoolStr),
    Optional('field on threshold'): Coerce(float),
    Optional('probe ref delay'): Coerce(float),
    Optional('lo width'): Coerce(float),
    Optional('dc width'): Coerce(float),
    Optional('detection axis zero pad to'): Coerce(int),
    Optional('gaussian power'): Coerce(float),
    Optional('pump chop'): Coerce(BoolStr),
    Optional('detrend t1'): Coerce(BoolStr),
    }, extra=REMOVE_EXTRA)

schema_tg = Schema({
    Required('jobname'): str,
    Required('batch'): Coerce(int),
    Required('when'): str, # probably should validate to a valid date
    Required('wavelengths'): str,
    Required('analysis path'): str,
    Required('data path'): str,
    Optional('plot'): Coerce(BoolStr),
    Optional('force'): Coerce(BoolStr),
    Optional('stark'): Coerce(BoolStr),
    Optional('field on threshold'): Coerce(float),
    Optional('probe ref delay'): Coerce(float),
    Optional('lo width'): Coerce(float),
    Optional('dc width'): Coerce(float),
    Optional('detection axis zero pad to'): Coerce(int),
    Optional('gaussian power'): Coerce(float),
    Optional('pump chop'): Coerce(BoolStr),
    }, extra=REMOVE_EXTRA)

schema_phasing = Schema({
    Required('path'): str,
    Required('reference name'): str,
    Required('reference batch'): Coerce(int),
    Required('experiment name'): str,
    Required('experiment batch'): Coerce(int),
    Optional('copy'): Coerce(BoolStr),
    Optional('phasing guess'): Coerce(FloatList),
    Optional('excitation axis zero pad to'): Coerce(int),
    Optional('nsteps'): Coerce(int),
    Optional('nstep success'): Coerce(int),
    Optional('pixel range to fit'): Coerce(IntList), # check that lb < ub
    Optional('force'): Coerce(BoolStr),
    Optional('plot'): Coerce(BoolStr),
    Optional('phaselock wl'): Coerce(float),
    Optional('central wl'): Coerce(float),
    Optional('phasing t2'): Coerce(float),
    Optional('phase correct'): Coerce(float),
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
        'field on threshold': 0.2,
        'waveforms per table' : 40,
        'central wl' : 650,
        'phaselock wl'  : 650,
        'probe ref delay' : 850.,
        'lo width' : 200,
        'dc width' : 200,
        'detection axis zero pad to' : 2048,
        'gaussian power' : 2,
        'detrend t1': False,
        },
    'tg' : {
        'plot' : False,
        'force' : False,
        'stark' : False,
        'field on threshold': 0.2,
        'probe ref delay' : 850.,
        'lo width' : 200,
        'dc width' : 200,
        'detection axis zero pad to' : 2048,
        'gaussian power' : 2,
        },
    'phasing': {
        'force': False,
        'plot' : False,
        'copy': False,
        'phasing guess': '25, 850, 0',
        'zero pad to' : 2048,
        'nsteps' : 300,
        'nstep success' : 100,
        'pixel range to fit': '0, 1340',
        'excitation axis zero pad to' : 2048,
        'phaselock wl' : 650,
        'central wl' : 650,
        'phasing t2' : 10000,
        'phase correct' : 0,
        },
    }

def parse_config(path, which='all'):
    path = Path(path)
    log.debug('parse_config got `{!s}` as input file'.format(path))

    schemas = {
        '2d': schema_2d,
        'tg': schema_tg,
        'pump probe': schema_pp,
        'phasing': schema_phasing}

    if which == 'all':
        to_validate = schemas.items()
    else:
        try:
            to_validate = [(which, schemas[which])]
        except KeyError:
            s = 'schema `{!s}` does not exist.'.format(which)
            log.error(s)
            raise ValueError(s)

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
        cfg = as_dict(cfg)
        validated = OrderedDict()

        for key, schema in to_validate:
            log.debug('validating section [{!s}]'.format(key))
            validated[key] = schema(cfg[key])

        log.debug('final schema is:')
        for sec, val in validated.items():
            log.debug('  {!s}:'.format(sec))
            for subsec, subval in val.items():
                log.debug('    {!s}: {!s}'.format(subsec, repr(subval)))
    except configparser.Error as e:
        s = 'error while parsing file `{!s}`'.format(path)
        log.error(s)
        log.error(e)
        raise ValueError(s)
    except voluptuous.MultipleInvalid as e:
        s = 'error while validating file `{!s}`:'.format(path)
        log.error(e)
        raise ValueError(s)

    return validated
