"""
Project Name: Vertical Test Data Importer
File Name: ImportOldVerticalTestData.py
Author: Liu Ming
Created Time: March 6th, 2025
Description:
This script reads old vertical test data from .xls files in a specified folder, processes the data, and imports it into a MySQL database. It handles duplicate column names, missing values, and ensures that the data is inserted into uniquely named tables.
Preparation Before Running:
1. Ensure MySQL server is running and accessible.
2. Create a database named 'VerticalTest' in your MySQL server.
3. Update the 'config' dictionary with your MySQL server credentials.
4. Place the .xls files to be imported in the specified 'folder_path'.
5. Install required Python packages:
    - pandas
    - mysql-connector-python
How to Run:
1. Open a terminal or command prompt.
2. Navigate to the directory containing this script.
3. Run the script using Python:
    ```
    python ImportOldVerticalTestData.py
    ```
Change Log:
- [Date] - [Description of changes made]
"""


import os
import pandas as pd
import mysql.connector

# Database configuration, add character set
config = {
    'user': 'root',
    'password': '123456',
    'host': 'localhost',
    'database': 'VerticalTest',
}

# Folder path
folder_path = r'd:/mliu/ML_Learning/nidaqmx_learning/MySQL/VT_old'

# Get all xls files
xls_files = [f for f in os.listdir(folder_path) if f.endswith('.xls')]

def make_unique(column_names):
    seen = {}  # Used to record lowercase column names and their counts
    unique_names = []
    for i, col in enumerate(column_names):
        # Handle null or empty strings
        if pd.isnull(col) or col.strip() == '':
            col = f'column_{i+1}'
        else:
            col = col.strip()  # Remove leading and trailing spaces
        original_col = col
        col_lower = col.lower()  # Convert to lowercase for comparison
        if col_lower in seen:
            seen[col_lower] += 1
            col = f"{original_col}_{seen[col_lower]}"  # Add suffix
        else:
            seen[col_lower] = 0
        unique_names.append(col)
    return unique_names

try:
    # Create database connection
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()

    for xls_file in xls_files:
        file_path = os.path.join(folder_path, xls_file)
        
        # Read xls file, use utf-8 encoding
        df = pd.read_csv(file_path, header=None, sep='\t', encoding='GB2312', on_bad_lines='skip')
        
        # Print original shape, check rows and columns
        print(f"Original shape of file {xls_file}: {df.shape}")
        
        # Assume the first row is the header
        header_row = df.iloc[0].tolist()
        print(f"Original header of file {xls_file}: {header_row}")
        
        # Handle duplicate column names
        unique_header = make_unique(header_row)
        print(f"Unique header of file {xls_file}: {unique_header}")
        
        # Set DataFrame column names
        df.columns = unique_header
        df = df[1:]  # Remove header row
        df = df.dropna(how='all')  # Remove rows that are completely empty
        
        # Print processed shape
        print(f"Processed shape of file {xls_file}: {df.shape}")
        
        # Replace NaN with None to insert MySQL NULL
        df = df.where(pd.notnull(df), None)
        
        # Remove file extension for table name
        table_name = os.path.splitext(xls_file)[0]

        # Check if table exists
        cursor.execute("SHOW TABLES LIKE %s", (table_name,))
        result = cursor.fetchone()
        if result:
            print(f"Table `{table_name}` already exists, skipping file `{xls_file}`")
            continue

        # Create table, specify utf8mb4 character set
        create_table_query = f'''
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            {', '.join([f'`{col}` VARCHAR(255)' for col in unique_header])}
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
        '''
        cursor.execute(create_table_query)
        print(f"Table `{table_name}` created successfully!")

        # Insert data
        insert_query = f'''
        INSERT INTO `{table_name}` ({', '.join([f'`{col}`' for col in unique_header])})
        VALUES ({', '.join(['%s'] * len(unique_header))})
        '''
        for index, row in df.iterrows():
            cursor.execute(insert_query, tuple(row))
        
        cnx.commit()
        print(f"Data from file `{xls_file}` imported to table `{table_name}` successfully!")

except mysql.connector.Error as err:
    print(f"Database error occurred: {err}")
except Exception as e:
    print(f"Other error occurred: {e}")
finally:
    cursor.close()
    cnx.close()