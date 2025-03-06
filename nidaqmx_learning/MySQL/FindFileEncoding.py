"""
Project Name: ML_Learning
File Name: FindFileEncoding.py
Author: Liu Ming
Created Time: March 6th, 2025
Description:
This script detects the encoding of a specified file using the chardet library.
Preparation:
1. Ensure the chardet library is installed. You can install it using pip:
    pip install chardet
2. Update the 'file_path' variable with the path to the file you want to check.
Usage:
Run the script using Python:
    python FindFileEncoding.py
Change Log:
March 6th, 2025 - Liu Ming - Initial creation.
"""

import chardet

file_path = r'd:/mliu/ML_Learning/nidaqmx_learning/MySQL/VT_old/20250228 PAPS-1300-S13-4KQ0Eacc_manual.xls'

with open(file_path, 'rb') as f:
    result = chardet.detect(f.read())
    print(f"File encoding: {result['encoding']}")