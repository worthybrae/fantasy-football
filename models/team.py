from helpers.database import execute_sql_file

class Team:

    def __init__(self, name, abbreviation):
        self.name = name.lower().strip()
        self.abbreviation = abbreviation.lower().strip()

    def check_if_exists(self, connection):
        variables = {
            'abbreviation': self.abbreviation
        }
        results = execute_sql_file(connection,'check_team.sql', variables)
        print(results)
        if len(results) > 0:
            return True
        else:
            return self.insert(connection)
        
    def insert(self, connection):
        try:
            variables = {
                'abbreviation': self.abbreviation,
                'name': self.name
            }
            execute_sql_file(connection,'insert_team.sql', variables, True)
            print(f"{self.name} ({self.abbreviation}) added to teams!")
            return True
        except Exception as e:
            print(f"Error during team insert: {e}")
            return False




