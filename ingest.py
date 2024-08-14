import requests
from bs4 import BeautifulSoup
import pandas as pd


# Function to scrape data for a given week
def get_game_urls(week):
    url = f"https://www.pro-football-reference.com/years/2023/week_{week}.htm"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    game_summaries = soup.find_all('div', class_='game_summary')
    
    game_urls = []

    for game in game_summaries:
        a_tags = game.find_all('a')
        for a in a_tags:
            if a['href'].startswith('/boxscores'):
                game_urls.append(f"https://www.pro-football-reference.com{a['href']}")
    return game_urls      

# Function to scrape data from a game URL
def scrape_game_data(game_url):
    response = requests.get(game_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Example: Extracting the game title and score (you can customize this)
    cells = soup.find_all('td', class_='right')
        
    for cell in cells:
        # Get all cells (td elements) in the row
        print(cell)
    
for week in range(1, 3):
    game_urls = get_game_urls(week)
    for game_url in game_urls:
        scrape_game_data(game_url)
        break
    break