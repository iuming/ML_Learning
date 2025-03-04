"""
Project: MySQL Database GUI
File: Database_GUI.py
Author: Liu Ming
Created: March 4th, 2025

This script creates a simple GUI application using Tkinter to view data from a MySQL database.
Modules:
    tkinter: Provides classes for creating GUI applications.
    mysql.connector: Provides classes for connecting to and interacting with a MySQL database.
Functions:
    fetch_data(): Connects to the MySQL database, retrieves data from the 'Test' table, and returns it.
Classes:
    DatabaseGUI: A class that creates the GUI application for viewing data from the MySQL database.
GUI Elements:
    root: The main window of the application.
    tree: A Treeview widget for displaying the data in a table format.
    scrollbar: A scrollbar for the Treeview widget.
Usage:
    Run this script to open the GUI application. The data from the 'Test' table in the MySQL database will be displayed.

Modification Log:
- March 4th, 2025: Initial creation by Liu Ming
- YYYY-MM-DD: [Modification Notes]
"""
import mysql.connector
import tkinter as tk
from tkinter import ttk
from datetime import datetime

# Database configuration
config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "VerticalTest"
}

# Function to fetch data
def fetch_data():
    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Test")
        rows = cursor.fetchall()
        return rows
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []
    finally:
        if conn:
            conn.close()

# GUI application
class DatabaseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VerticalTest Database Viewer")
        self.root.geometry("1200x600")

        # Create table
        self.tree = ttk.Treeview(self.root, columns=self.get_columns(), show="headings")
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Add scrollbar
        self.scrollbar = ttk.Scrollbar(self.tree, orient="vertical", command=self.tree.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=self.scrollbar.set)

        # Add table headers
        columns = self.get_columns()
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        # Load data
        self.load_data()

    def get_columns(self):
        # Table headers (assuming table name and column names are known)
        return [
            "id", "Pin", "Pr", "Pt", "Q0", "Eacc", "Qin", "Qt", 
            "QL", "Q0_initial", "R_Q", "Leff", "Ep_Eacc", "Bp_Eacc", 
            "Time", "Radiation1", "Radiation2", "Radiation3", 
            "Temperature1", "Temperature2", "Temperature3"
        ]

    def load_data(self):
        # Clear existing data
        for row in self.tree.get_children():
            self.tree.delete(row)

        # Fetch data
        rows = fetch_data()
        for row in rows:
            # Format Time column as string
            formatted_time = row[13].strftime("%Y-%m-%d %H:%M:%S")
            self.tree.insert("", "end", values=row[:13] + (formatted_time,) + row[14:])

# Main function
if __name__ == "__main__":
    root = tk.Tk()
    app = DatabaseGUI(root)
    root.mainloop()
