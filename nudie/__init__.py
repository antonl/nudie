from __future__ import division, absolute_import, unicode_literals, print_function
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

from .utils.winspec import SpeFile
from .utils.paths import setup_paths
from .utils.batch_loader import load_job

try:
    mount_point = setup_paths()
except RuntimeError as e:
    pass




