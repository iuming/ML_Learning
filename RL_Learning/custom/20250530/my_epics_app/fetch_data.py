import epics
import time
import os

# Set EPICS Channel Access environment variables
os.environ['EPICS_CA_ADDR_LIST'] = "192.168.246.128"
# os.environ['EPICS_CA_AUTO_ADDR_LIST'] = "NO"

# Connect to the temperature reading PV
temp_pv = epics.PV("TEMP:READING")
control_pv = epics.PV("TEMP:CONTROL")

# # Test reading and writing
# print("Monitoring temperature for 10 seconds...")
# for _ in range(10):
#     temp = temp_pv.get()
#     print(f"Temperature: {temp:.1f}°C")
#     # Simulate control (optional)
#     control_pv.put(1)  # Increase temperature
#     time.sleep(1)

print("Done.")