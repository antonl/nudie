import nudie
from nudie.dataset import (LinearStark, FastDazzlerSetupLoader, Config,
                           WavelengthCalibration, AttachWavelengthAxis, PP)

nudie.show_errors(nudie.logging.INFO)

'''
d = Dataset('test.h5', 'w')
print('readonly? {!s}'.format(d.readonly))
try:
    d = Dataset('test2.h5', 'r')
    print('readonly? {!s}'.format(d.readonly))
except Exception:
    pass
PP('pp-phasing-batch00.h5')
DD('2d-phasing-batch00.h5')
DDESS('2dess-batch00.h5')
TG('tg-batch00.h5')
TGStark('tg-stark-batch00.h5')

'''
'''
c = Config(
    {'FastDazzlerSetupLoader': 
        {'batch_path': 'C:/aloukian-project/16-10-10/test-data/tips-3mp-gly-77k-linear-stark-batch00',
         'batch_name': 'tips-3mp-gly-77k-linear-stark',
         'data_type': LinearStark,
         'allow_parallel': False,
        }})
loader = FastDazzlerSetupLoader(config=c)
wl = WavelengthCalibration.from_mat_file('C:/aloukian-project/16-10-10/calib-spec-600g-605nm.mat')
xfm = AttachWavelengthAxis(wl)
a = loader.load()
res = a.transform(xfm, dim=1).compute()

res.path = 'C:/aloukian-project/16-10-10/tips-3mp-gly-77k-linear-stark.h5'
'''
#res.save(overwrite=True)

#res = a

#import matplotlib.pyplot as plt
#plt.plot(res.axes[1].data, res.stark_spectra[0])
#plt.tight_layout()
#plt.show()
c = Config(
    {'FastDazzlerSetupLoader': 
        {'batch_path': 'C:/aloukian-project/16-10-10/test-data/tips-3mp-gly-77k-pp-10ps-phasing-batch02',
         'batch_name': 'tips-3mp-gly-77k-pp-10ps-phasing',
         'data_type': PP,
         'allow_parallel': False,
        }})
loader = FastDazzlerSetupLoader(config=c)
wl = WavelengthCalibration.from_mat_file('C:/aloukian-project/16-10-10/calib-spec-600g-605nm.mat')
xfm = AttachWavelengthAxis(wl)
a = loader.load()
res = a.transform(xfm, dim=1).compute()
#res = a.compute()

import matplotlib.pyplot as plt
plt.plot(res.axes[1].data, res.pump_probe_spectra[0])
plt.tight_layout()
plt.show()

