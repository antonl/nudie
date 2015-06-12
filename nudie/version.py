version = '0.2.3-dev'

import re

def parse_version(version):
    reg = re.compile("(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(\-(?P<release>[a-z]+))?")
    tmp = reg.match(version)

    version_tuple_rel = (int(tmp.group('major')), int(tmp.group('minor')),
            int(tmp.group('patch')), tmp.group('release'))

    version_tuple = version_tuple_rel[:3]
    return version_tuple, version_tuple_rel
