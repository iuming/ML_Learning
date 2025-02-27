import nidaqmx
import pprint
import numpy as np 
from matplotlib import pyplot as plt 

pp = pprint.PrettyPrinter(indent=4)


with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("cDAQ9189-1D712C2Mod1/ai0")

    print('1 Channel 1 Sample Read: ')
    data = task.read()
    pp.pprint(data)

    data = task.read(number_of_samples_per_channel=1)
    pp.pprint(data)

    print('1 Channel N Samples Read: ')
    data = task.read(number_of_samples_per_channel=10)
    x=np.arange(0,len(data))
    pp.pprint(data)
    plt.plot(x,data)

    task.ai_channels.add_ai_voltage_chan("cDAQ9189-1D712C2Mod1/ai1")

    print('N Channel 1 Sample Read: ')
    data = task.read()
    pp.pprint(data)

    print('N Channel N Samples Read: ')
    data = task.read(number_of_samples_per_channel=2)
    pp.pprint(data)
