"""
Project Name: ML_Learning
File Name: Create_database.py
Author: mliu
Created Time: March 4th, 2025
Description:
This script connects to a MySQL server and creates a new database with the specified name.
Prerequisites:
- MySQL server should be installed and running.
- MySQL connector for Python should be installed (mysql-connector-python).
How to Run:
1. Ensure MySQL server is running.
2. Update the connection parameters (host, user, password) if necessary.
3. Run the script using Python: `python Create_database.py`
Modification Log:
- Motified Time: March 4th, 2025
- Motified By: mliu
    Motified Notes: Initial creation of the script.
"""


import mysql.connector
from mysql.connector import Error

def create_database(database_name):
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456"
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute(f"CREATE DATABASE {database_name}")
            print(f"数据库 '{database_name}' 创建成功")
    
    except Error as e:
        print(f"错误: {e}")
    
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()

create_database("my_database")