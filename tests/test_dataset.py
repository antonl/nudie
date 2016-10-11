import nudie
from nudie.dataset import (LinearStark, FastDazzlerSetupLoader, Config,
                           WavelengthCalibration)

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
c = Config(
    {'FastDazzlerSetupLoader': 
        {'batch_path': 'C:/aloukian-project/16-10-10/test-data/tips-3mp-gly-77k-linear-stark-batch00',
         'batch_name': 'tips-3mp-gly-77k-linear-stark',
         'data_type': LinearStark,
         'allow_parallel': True,
        }})
loader = FastDazzlerSetupLoader(config=c)

a = loader.load()
res = a.compute()

#res = a

#import matplotlib.pyplot as plt
#plt.plot(res.stark_spectra[0])
#plt.tight_layout()
#plt.show()

res.path = 'C:/aloukian-project/16-10-10/tips-3mp-gly-77k-linear-stark.h5'
res.save(overwrite=True)

wl = WavelengthCalibration.from_mat_file('C:/aloukian-project/16-10-10/calib-spec-600g-605nm.mat')
wl.save(overwrite=True)
wl2 = WavelengthCalibration()
wl2.load('C:/aloukian-project/16-10-10/calib-spec-600g-605nm.h5')
'''

from traitlets.config import Config

c = Config(
    {'RawData': 
        {'phase_cycles': ['none1', 'zerozero', 'none2', 'pipi'],
         'batch_path': 'test-data/clh-linear-stark-s3-batch00',
         'batch_name': 'clh-linear-stark-s3'
        }})
#graph = delayed(RawData)(config=c)
#print(graph.process().compute().analog_channels)

RawData(config=c).process()
'''
