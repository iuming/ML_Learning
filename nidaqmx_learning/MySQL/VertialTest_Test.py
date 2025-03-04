'''
Project Name: MySQL Vertical Test
File Name: VertialTest_Test.py
Author: Liu Ming
Created Date: March 4th, 2025

This script connects to a MySQL database and performs the following operations:
1. Creates a database named 'VerticalTest' if it does not already exist.
2. Creates a table named 'Test' within the 'VerticalTest' database if it does not already exist.
3. Generates random test data and inserts 10,000 rows into the 'Test' table.
Functions:
- create_database(): Connects to the MySQL server and creates the 'VerticalTest' database if it does not exist.
- create_table(): Connects to the 'VerticalTest' database and creates the 'Test' table if it does not exist.
- generate_test_data(): Generates a dictionary containing random test data for the 'Test' table.
- insert_test_data(): Generates and inserts 10,000 rows of random test data into the 'Test' table.
Configuration:
- The database connection configuration is stored in the 'config' dictionary.
Usage:
- Run the script directly to create the database, create the table, and insert the test data.

Modification Log:
- March 4th, 2025: Initial creation by Liu Ming
- YYYY-MM-DD: [Modification Notes]
'''


import mysql.connector

import random
import datetime

# Database configuration
config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "VerticalTest"
}

# Create database
def create_database():
    conn = mysql.connector.connect(
        host=config["host"],
        user=config["user"],
        password=config["password"]
    )
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES LIKE 'VerticalTest'")
    result = cursor.fetchone()
    if not result:
        cursor.execute("CREATE DATABASE VerticalTest")
        print("Database 'VerticalTest' created successfully.")
    else:
        print("Database 'VerticalTest' already exists.")
    cursor.close()
    conn.close()

# Create table
def create_table():
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Test (
        id INT AUTO_INCREMENT PRIMARY KEY,
        Pin FLOAT,
        Pr FLOAT,
        Pt FLOAT,
        Q0 FLOAT,
        Eacc FLOAT,
        Qin FLOAT,
        Qt FLOAT,
        QL FLOAT,
        Q0_initial FLOAT,
        R_Q FLOAT,
        Leff FLOAT,
        Ep_Eacc FLOAT,
        Bp_Eacc FLOAT,
        Time DATETIME,
        Radiation1 FLOAT,
        Radiation2 FLOAT,
        Radiation3 FLOAT,
        Temperature1 FLOAT,
        Temperature2 FLOAT,
        Temperature3 FLOAT
    )
    """)
    print("Table 'Test' created successfully.")
    cursor.close()
    conn.close()

# Generate random test data
def generate_test_data():
    data = {
        "Pin": random.uniform(0, 100),
        "Pr": random.uniform(0, 100),
        "Pt": random.uniform(0, 100),
        "Q0": random.uniform(0, 100),
        "Eacc": random.uniform(0, 100),
        "Qin": random.uniform(0, 100),
        "Qt": random.uniform(0, 100),
        "QL": random.uniform(0, 100),
        "Q0_initial": random.uniform(0, 100),
        "R_Q": random.uniform(0, 100),
        "Leff": random.uniform(0, 100),
        "Ep_Eacc": random.uniform(0, 100),
        "Bp_Eacc": random.uniform(0, 100),
        "Time": datetime.datetime.now(),
        "Radiation1": random.uniform(0, 100),
        "Radiation2": random.uniform(0, 100),
        "Radiation3": random.uniform(0, 100),
        "Temperature1": random.uniform(0, 100),
        "Temperature2": random.uniform(0, 100),
        "Temperature3": random.uniform(0, 100)
    }
    return data

def insert_test_data():
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    insert_query = """
    INSERT INTO Test (
        Pin, Pr, Pt, Q0, Eacc, Qin, Qt, QL,
        Q0_initial, R_Q, Leff, Ep_Eacc, Bp_Eacc,
        Time, Radiation1, Radiation2, Radiation3,
        Temperature1, Temperature2, Temperature3
    )
    VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s, %s
    )
    """
    # Generate and insert 10000 rows of data
    data = []
    for _ in range(10000):
        row = generate_test_data()
        data.append((
            row["Pin"], row["Pr"], row["Pt"], row["Q0"], row["Eacc"],
            row["Qin"], row["Qt"], row["QL"], row["Q0_initial"],
            row["R_Q"], row["Leff"], row["Ep_Eacc"], row["Bp_Eacc"],
            row["Time"], row["Radiation1"], row["Radiation2"],
            row["Radiation3"], row["Temperature1"], row["Temperature2"],
            row["Temperature3"]
        ))
    # Batch insert data
    cursor.executemany(insert_query, data)
    conn.commit()
    print(f"Inserted {len(data)} rows into the Test table.")
    cursor.close()
    conn.close()

# Main function
if __name__ == "__main__":
    try:
        create_database()
        create_table()
        insert_test_data()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
