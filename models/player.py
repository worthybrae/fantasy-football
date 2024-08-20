from helpers.database import execute_sql_file
import requests
from bs4 import BeautifulSoup


class Player:

    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.position = None
        self.born = None

    def check_if_exists(self, connection):
        variables = {
            'id': self.id
        }
        results = execute_sql_file(connection,'check_player.sql', variables)
        if len(results) > 0:
            self.position = results[0][0]
            self.born = results[0][1]
            return True
        else:
            if self.get_player():
                return self.insert(connection)
        
    def get_player(self):
        # URL of the webpage
        url = f"https://www.pro-football-reference.com/players/{self.id[0]}/{self.id}.htm"

        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the div with id "info"
            info_div = soup.find('div', id='info')
            
            if info_div:
                # Find all p elements within the info div
                p_elements = info_div.find_all('p')
                
                # Iterate through the p elements to find the desired information
                for p in p_elements:
                    if 'Position:' in p.text:
                        self.position = p.text.split(':')[1].strip().lower()
                    elif 'Born:' in p.text:
                        self.born = p.text.split(':')[1].split('in')[0].split(',')[1].strip()
                        break
                return True
                
            else:
                print("Could not find the div with id 'info'")
                return False
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
            return False
        
    def insert(self, connection):
        try:
            variables = {
                'id': self.id,
                'name': self.name,
                'position': self.position,
                'born': self.born
            }
            execute_sql_file(connection,'insert_player.sql', variables, True)
            print(f"{self.name} ({self.id}) added to players!")
            return True
        except Exception as e:
            print(f"Error during team insert: {e}")
            return False