from helpers.database import create_connection, execute_sql_file

def main():
    # Create a database connection
    connection = create_connection()

    if connection is not None:
        # List of SQL files to execute
        sql_files = [
            "create_team.sql",
            "create_player.sql",
            "create_game.sql",
            "create_snaps.sql",
            "create_rushing.sql",
            "create_receiving.sql",
            "create_passing.sql"
        ]

        # Execute each SQL file
        for sql_file in sql_files:
            print(sql_file)
            execute_sql_file(connection, sql_file)
        
        # Commit the changes and close the connection
        connection.commit()
        connection.close()
        print("All tables created successfully.")
    else:
        print("Error! Cannot create the database connection.")

if __name__ == "__main__":
    main()

