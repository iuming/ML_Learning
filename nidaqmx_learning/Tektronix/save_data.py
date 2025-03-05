from tm_devices import DeviceManager
import time
import csv

with DeviceManager(verbose=True) as dm:
    scope = dm.add_scope("10.4.193.26")  # Replace with your oscilloscope’s IP
    scope.visa_resource.timeout = 10000  # 10-second timeout

    # Configure oscilloscope (example settings)
    scope.write(":WAVEFORM:SOURCE CH4")  # Set source to Channel 1
    scope.write(":WAVEFORM:FORMAT BYTE")  # Use byte format for binary data

    # Wait for trigger (adjust condition as needed based on your intent)
    print("Waiting for oscilloscope to trigger...")
    while scope.query(":TRIGGER:STATE?").strip() not in ["TRIGGERED", "STOPPED"]:
        time.sleep(0.1)
    print("Oscilloscope triggered successfully")

    # Retrieve waveform data using visa_resource
    waveform_data = scope.visa_resource.query_binary_values(":WAVEFORM:DATA?", datatype="b", is_big_endian=False)

    # Save to CSV
    with open("waveform_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Sample", "Voltage"])
        for i, value in enumerate(waveform_data):
            writer.writerow([i, value])
    print("Waveform data saved to waveform_data.csv")