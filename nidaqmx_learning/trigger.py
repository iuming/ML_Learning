import nidaqmx
from nidaqmx.constants import Edge, AcquisitionType
import numpy as np
import pandas as pd

# Create a task
with nidaqmx.Task() as task:
    # Add an analog input voltage channel
    task.ai_channels.add_ai_voltage_chan("cDAQ9189-1D712C2Mod1/ai0")
    
    # Configure the sample clock
    sample_rate = 500000
    samples_to_read = sample_rate  # 1 second of data
    
    # Configure the trigger
    task.triggers.start_trigger.cfg_anlg_edge_start_trig(
        trigger_source="cDAQ9189-1D712C2Mod1/ai0",
        trigger_slope=Edge.RISING,
        trigger_level=0.9
    )
    
    # Configure the timing
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=samples_to_read
    )
    
    # Start the task and read the data
    data = task.read(number_of_samples_per_channel=samples_to_read)
    
    # Save the data to a CSV file
    df = pd.DataFrame(data)
    df.to_csv('trigger.csv', index=False, header=False)