from tm_devices import DeviceManager
from tm_devices.drivers import MSO6
from pathlib import Path

EXAMPLE_CSV_FILE = Path("example_curve_query.csv")

with DeviceManager(verbose=True) as dm:
    scope: MSO6 = dm.add_scope("10.4.193.26")
    # afg: AFG3KC = dm.add_afg("10.4.193.26")
    # scope = dm.add_scope("10.4.193.26")

    # Turn on AFG
    # afg.set_and_check(":OUTPUT1:STATE", "1")

    # Perform curve query and save results to csv file
    curve_returned = scope.curve_query(1, output_csv_file=EXAMPLE_CSV_FILE)

# Read in the curve query from file
with EXAMPLE_CSV_FILE.open(encoding="utf-8") as csv_content:
    curve_saved = [int(i) for i in csv_content.read().split(",")]

# Verify query saved to csv is the same as the one returned from curve_query function call
assert curve_saved == curve_returned