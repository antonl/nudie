import nudie
from nudie.dataset import *
from dask.delayed import delayed

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
        {'batch_path': 'test-data/clh-linear-stark-s3-batch00',
         'batch_name': 'clh-linear-stark-s3',
         'data_type': LinearStark,
         'allow_parallel': True,
        }})
loader = FastDazzlerSetupLoader(config=c)

a = loader.load()
res = a.compute()

#res = a

import matplotlib.pyplot
matplotlib.pyplot.plot(res.stark_spectra[0])
matplotlib.pyplot.show()

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
