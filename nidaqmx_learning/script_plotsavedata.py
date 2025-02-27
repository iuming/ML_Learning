"""
Project Name: NI-DAQmx Learning
File Name: script_plotsavedata.py
Author: Liu Ming
Created Time: Feb 27th, 2025
Description:
This script reads a CSV file containing voltage data from multiple channels, 
plots the data for each channel, and displays the plot. The CSV file name can 
be provided as a command line argument. If no file name is provided, a default 
file 'nidaqmx_learning/test.csv' is used.
Preparation:
- Ensure that the required CSV file with voltage data is available.
- Install the necessary Python packages: pandas and matplotlib.
Run Instructions:
- To run the script with a specific CSV file:
    python script_plotsavedata.py <path_to_csv_file>
- To run the script with the default CSV file:
    python script_plotsavedata.py
Change Log:
- Modified Time: [Add modification date]
    Modified Notes: [Add modification details]
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys

# Get the file name from command line arguments, use default 'data.csv' if not provided
if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    file_name = 'nidaqmx_learning/test.csv'

try:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_name)
    
    # Check if the DataFrame is empty
    if df.empty:
        print("The CSV file has no data rows.")
    else:
        # Plot the data for each column
        for column in df.columns:
            plt.plot(df.index, df[column], label=column)
        
        # Set axis labels and title
        plt.xlabel('Sample Index')
        plt.ylabel('Voltage')
        plt.title('Channel Data Plot')
        
        # Add legend
        plt.legend()
        
        # Show the plot
        plt.show()

# Handle possible exceptions
except pd.errors.EmptyDataError:
    print("The CSV file is empty.")
except FileNotFoundError:
    print(f"The file '{file_name}' was not found.")
except pd.errors.ParserError:
    print("Error parsing the CSV file.")
except Exception as e:
    print(f"An error occurred: {e}")
