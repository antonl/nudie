from __future__ import division, absolute_import, unicode_literals, print_function
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

def show_errors(level=logging.WARNING):
    sh = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s') 
    sh.setFormatter(fmt)
    log.handlers.pop().close() # close the last handler
    log.addHandler(sh)
    log.setLevel(level)

from . import utils
from .spectrometer import simple_wavelength_axis, wavelen_to_freq, \
    freq_to_wavelen
from .utils.winspec import SpeFile
from .utils.paths import setup_paths
from .version import version, parse_version
from .utils.analysis_bits import tag_phases, cleanup_analogtxt, \
    detect_table_start, synchronize_daq_to_camera, load_analogtxt, \
    determine_shutter_shots, load_camera_file, trim_all, \
    make_6phase_cycler, select_prd_peak, remove_incomplete_t1_waveforms

from .schema import parse_config
from pathlib import Path

try:
    mount_point, data_folder = setup_paths()
except RuntimeError as e:
    pass
finally:
    mount_point = Path('.')
    data_folder = Path('.')

# this must be loaded after setup_paths is run
from .utils.batch_loader import load_job

from . import dd, pump_probe, phasing, apply_phase, stark_dd, dataset

