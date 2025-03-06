"""
Project Name: Database Query Tool for Maximum Eacc
File Name: FindMaxEacc_GUI.py
Author: Liu Ming
Created: March 6th, 2025
Description:
This script creates a GUI application using Tkinter to connect to a MySQL database, 
retrieve the table with the maximum Eacc (MV/m) value, and display the relevant information 
in a user-friendly interface. The application allows users to find the maximum Eacc value 
across all tables in the specified database and displays details such as test date, cavity type, 
cavity ID, and test temperature.
Prerequisites:
- MySQL server running with the specified database and tables.
- Python 3.x installed.
- Required Python packages: tkinter, mysql-connector-python.
How to Run:
1. Ensure MySQL server is running and accessible with the provided credentials.
2. Install required packages using pip:
    pip install mysql-connector-python
3. Run the script:
    python FindMaxEacc_GUI.py
Changelog:
- March 6th, 2025 Liu Ming - Initial creation of the script.
- March 6th, 2025 Liu Ming - Improved GUI layout and styling.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import mysql.connector

# Database connection configuration
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "VerticalTest"
}

def get_max_eacc_table():
    try:
        # Connect to the database
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()

        # Get all table names
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]

        max_eacc = -1
        max_table_name = ""

        # Iterate through each table to find the table with the maximum Eacc(MV/m)
        for table in tables:
            # Check if the table has the Eacc(MV/m) column
            cursor.execute(f"SHOW COLUMNS FROM `{table}` LIKE 'Eacc(MV/m)'")
            if cursor.fetchone():
                cursor.execute(f"SELECT MAX(CAST(`Eacc(MV/m)` AS UNSIGNED)) FROM `{table}`")
                result = cursor.fetchone()
                if result and result[0] is not None:
                    # Convert the result to a float
                    current_max_eacc = float(result[0])
                    if current_max_eacc > max_eacc:
                        max_eacc = current_max_eacc
                        max_table_name = table

        cursor.close()
        connection.close()

        return max_table_name, max_eacc

    except mysql.connector.Error as error:
        messagebox.showerror("Database Error", f"Failed to connect to the database: {error}")
        return None, None

def show_max_eacc_table():
    max_table, max_eacc = get_max_eacc_table()
    if max_table and max_eacc is not None:
        # Parse table name
        table_parts = max_table.split("-")
        if len(table_parts) >= 4:
            test_date = table_parts[0]
            cavity_type = table_parts[1].replace("mhz", "MHz")
            cavity_id = table_parts[2]
            test_temp = table_parts[3].replace("k", "K")
        else:
            test_date = "Unknown"
            cavity_type = "Unknown"
            cavity_id = "Unknown"
            test_temp = "Unknown"

        # Update display content
        result_frame.pack_forget()
        result_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        # Test date frame
        test_date_frame = ttk.Frame(result_frame, borderwidth=2, relief=tk.GROOVE)
        test_date_frame.pack(fill=tk.X, padx=10, pady=5)
        test_date_label = ttk.Label(test_date_frame, text="Test Date:", font=('Times New Roman', 14, 'bold'))
        test_date_label.pack(side=tk.LEFT, padx=10)
        test_date_value = ttk.Label(test_date_frame, text=test_date, font=('Times New Roman', 56))
        test_date_value.pack(side=tk.LEFT, padx=10)

        # Cavity type frame
        cavity_type_frame = ttk.Frame(result_frame, borderwidth=2, relief=tk.GROOVE)
        cavity_type_frame.pack(fill=tk.X, padx=10, pady=5)
        cavity_type_label = ttk.Label(cavity_type_frame, text="Cavity Type:", font=('Times New Roman', 14, 'bold'))
        cavity_type_label.pack(side=tk.LEFT, padx=10)
        cavity_type_value = ttk.Label(cavity_type_frame, text=cavity_type, font=('Times New Roman', 56))
        cavity_type_value.pack(side=tk.LEFT, padx=10)

        # Cavity ID frame
        cavity_id_frame = ttk.Frame(result_frame, borderwidth=2, relief=tk.GROOVE)
        cavity_id_frame.pack(fill=tk.X, padx=10, pady=5)
        cavity_id_label = ttk.Label(cavity_id_frame, text="Cavity ID:", font=('Times New Roman', 14, 'bold'))
        cavity_id_label.pack(side=tk.LEFT, padx=10)
        cavity_id_value = ttk.Label(cavity_id_frame, text=cavity_id, font=('Times New Roman', 56))
        cavity_id_value.pack(side=tk.LEFT, padx=10)

        # Test temperature frame
        test_temp_frame = ttk.Frame(result_frame, borderwidth=2, relief=tk.GROOVE)
        test_temp_frame.pack(fill=tk.X, padx=10, pady=5)
        test_temp_label = ttk.Label(test_temp_frame, text="Test Temperature:", font=('Times New Roman', 14, 'bold'))
        test_temp_label.pack(side=tk.LEFT, padx=10)
        test_temp_value = ttk.Label(test_temp_frame, text=test_temp, font=('Times New Roman', 56))
        test_temp_value.pack(side=tk.LEFT, padx=10)

        # Maximum gradient frame
        max_eacc_frame = ttk.Frame(result_frame, borderwidth=2, relief=tk.GROOVE)
        max_eacc_frame.pack(fill=tk.X, padx=10, pady=5)
        max_eacc_label = ttk.Label(max_eacc_frame, text="Maximum Gradient:", font=('Times New Roman', 14, 'bold'))
        max_eacc_label.pack(side=tk.LEFT, padx=10)
        max_eacc_value = ttk.Label(max_eacc_frame, text=f"{max_eacc} MV/m", font=('Times New Roman', 84))
        max_eacc_value.pack(side=tk.LEFT, padx=10)
    else:
        result_frame.pack_forget()
        result_label.config(text="No table found with Eacc(MV/m) column")
        result_label.pack()

# Create tkinter window
root = tk.Tk()
root.title("Database Query Tool")
root.geometry("1000x900")

# Use ttk.Style to set styles
style = ttk.Style()
style.configure("TButton", padding=10, font=('Times New Roman', 14))
style.configure("TLabel", padding=10, font=('Times New Roman', 14))

# Create main frame
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Create button frame
button_frame = ttk.Frame(main_frame)
button_frame.pack(fill=tk.X, pady=20)

# Create result display frame
result_frame = ttk.Frame(main_frame)
result_frame.pack(fill=tk.BOTH, expand=True)

# Create button
search_button = ttk.Button(button_frame, text="Find Maximum Eacc in History", command=show_max_eacc_table)
search_button.pack(fill=tk.X)

# Create default result display label
result_label = ttk.Label(result_frame, text="", anchor=tk.CENTER)
result_label.pack(fill=tk.BOTH, expand=True)

# Run tkinter main loop
root.mainloop()