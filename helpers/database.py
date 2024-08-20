import sqlite3
import re


def create_connection():
    """Create and return a connection to an SQLite database."""
    try:
        connection = sqlite3.connect('database/stats.db')
        print(f"Connection to SQLite DB 'database/stats.db' successful")
        return connection
    except sqlite3.Error as e:
        print(f"Error: '{e}' occurred")
        return None

def execute_sql_file(connection, sql_file, variables=None, commit=False):
    """Read and execute an SQL file."""
    # Read query file
    with open(f"queries/{sql_file}", 'r') as file:
        sql_script = file.read()
    
    # Add variables to query
    if variables is not None:
        for variable, value in variables.items():
            pattern = r'\$\{' + re.escape(variable) + r'\}'
            sql_script = re.sub(pattern, str(value), sql_script)

    # Execute query
    try:
        cursor = connection.cursor()
        cursor.execute(sql_script)
        if commit:
            connection.commit()
        print(f"Executed {sql_file} successfully.")
        # Fetch all results
        results = cursor.fetchall()
        cursor.close()
        return results
    except sqlite3.Error as e:
        print(f"Error executing {sql_file}: '{e}'")
        cursor.close()
        return None
        