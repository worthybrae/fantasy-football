import sqlite3

def create_connection():
    """Create and return a connection to an SQLite database."""
    try:
        connection = sqlite3.connect('database/stats.db')
        print(f"Connection to SQLite DB 'database/stats.db' successful")
        return connection
    except sqlite3.Error as e:
        print(f"Error: '{e}' occurred")
        return None

def execute_sql_file(connection, sql_file):
    """Read and execute an SQL file."""
    with open(f"queries/{sql_file}", 'r') as file:
        sql_script = file.read()
    
    try:
        cursor = connection.cursor()
        cursor.executescript(sql_script)
        print(f"Executed {sql_file} successfully.")
    except sqlite3.Error as e:
        print(f"Error executing {sql_file}: '{e}'")
    finally:
        cursor.close()