import requests
from bs4 import BeautifulSoup

# URL of the webpage
url = "https://www.pro-football-reference.com/players/E/EtieTr00.htm"

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
        
        # Initialize variables to store the desired information
        position = ""
        born = ""
        
        # Iterate through the p elements to find the desired information
        for p in p_elements:
            if 'Position:' in p.text:
                position = p.text.split(':')[1].strip().lower()
            elif 'Born:' in p.text:
                born = p.text.split(':')[1].split('in')[0].split(',')[1].strip()
        
        # Print the results
        print(position)
        print(born)
    else:
        print("Could not find the div with id 'info'")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")