"""
Project Name: ML_Learning
File Name: ImportCavityPropertiesData.py
Author: Liu Ming
Created: March 6th, 2025
Description:
This script reads cavity properties data from a CSV file, cleans the data, and inserts it into a MySQL database.
The script performs the following steps:
1. Reads the CSV file using pandas.
2. Cleans the data by removing rows containing NaN values and renaming columns.
3. Connects to a MySQL database.
4. Creates a table if it does not exist.
5. Inserts the cleaned data into the MySQL table.
Prerequisites:
- Ensure you have Python installed on your system.
- Install the required Python packages using the following command:
    pip install pandas mysql-connector-python
How to Run:
1. Update the `file_path` variable with the path to your CSV file.
2. Update the `config` dictionary with your MySQL database credentials.
3. Run the script using the following command:
    python ImportCavityPropertiesData.py
Adjustments:
- Modify the `file_path` variable to point to your CSV file.
- Update the `config` dictionary with your MySQL database credentials.
- Adjust the column names in the `df.columns` list if your CSV file has different column names.
"""

import pandas as pd

# Read CSV file
file_path = 'nidaqmx_learning/MySQL/CavityProperties.csv'  # Replace with your file path
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Display the first few rows to confirm data is read successfully
print(df.head())

# Data cleaning (remove rows containing NaN and rename columns)
df.columns = ['Cavity_Type', 'Frequency_MHz', 'Leff_nm', 'Ep_Ea', 'Bp_Ea_mT_MW', 'R_Q_Ohm', 'G_Ohm', 'Note']
df = df.dropna()

# Output cleaned data
print(df)

# Insert each row of parameters into MySQL database
import mysql.connector

# Database configuration
config = {
    'user': 'root',    # Replace with your MySQL username
    'password': '123456', # Replace with your MySQL password
    'host': 'localhost',         # Database host name
    'database': 'VerticalTest',   # Database name
}

try:
    # Create database connection
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()

    # SQL statement to create table
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS CavityPropertiesTable (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(50),
        frequency FLOAT,
        leff FLOAT,
        ep_ea FLOAT,
        bp_ea FLOAT,
        r_q FLOAT,
        g FLOAT,
        note VARCHAR(255)
    )
    '''
    
    # Execute SQL statement to create table
    cursor.execute(create_table_query)
    print("Table CavityPropertiesTable created successfully!")

    # SQL statement to insert data
    insert_query = '''
    INSERT INTO CavityPropertiesTable (name, frequency, leff, ep_ea, bp_ea, r_q, g, note) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    '''

    for index, row in df.iterrows():
        cursor.execute(insert_query, (
            row['Cavity_Type'],
            row['Frequency_MHz'],
            row['Leff_nm'],
            row['Ep_Ea'],
            row['Bp_Ea_mT_MW'],
            row['R_Q_Ohm'],
            row['G_Ohm'],
            row['Note']
        ))

    cnx.commit()  # Commit transaction
    print("Data inserted successfully!")

except mysql.connector.Error as err:
    print(f"Error occurred: {err}")
finally:
    cursor.close()
    cnx.close()