# File: football_data_extractor.py

import requests
from bs4 import BeautifulSoup
from helpers.browser import Browser
from helpers.parser import process_table


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
    print(f"Found {len(game_urls)} game urls for week {week}")
    return game_urls

def extract_football_data(url):
    with Browser(headless=True) as selenium:
        try:
            # Navigate to game url
            selenium.get_page(url)
            # Scroll until player offense stats table is present
            selenium.scroll_to_element("getElementById", "player_offense")
            # Process passing data
            offense_data = process_table(selenium.get_page_source(), "player_offense")
            # Scroll until advanced passing stats table is present
            selenium.scroll_to_element("getElementById", "passing_advanced")
            # Process passing data
            passing_data = process_table(selenium.get_page_source(), "passing_advanced")
            # Scroll until advanced rushing stats table is present
            selenium.scroll_to_element("getElementById", "rushing_advanced")
            # Process rushing data
            rushing_data = process_table(selenium.get_page_source(), "rushing_advanced")
            # Scroll until advanced receiving stats table is present
            selenium.scroll_to_element("getElementById", "receiving_advanced")
            # Process receiving data
            receiving_data = process_table(selenium.get_page_source(), "receiving_advanced")
            # Scroll until home snap counts table is present
            selenium.scroll_to_element("getElementById", "home_snap_counts")
            # Process home snap counts
            home_snap_counts = process_table(selenium.get_page_source(), "home_snap_counts")
            # Process away snap counts
            away_snaps_data = process_table(selenium.get_page_source(), "away_snap_counts")

        except Exception as e:
            print(f"An error occurred during data extraction: {e}")
            return None

def process_week(week):
    game_urls = get_game_urls(week)
    all_game_data = []

    for url in game_urls:
        print(f"Processing game: {url}")
        game_data = extract_football_data(url)
        if game_data:
            all_game_data.extend(game_data)
    
    return all_game_data

if __name__ == "__main__":
    season_data = {
        '2018': 17,
        '2019': 17,
        '2020': 17,
        '2021': 18,
        '2022': 18,
        '2023': 18,
    }
    
    for year, weeks in season_data.items():
        for week in range(1, weeks + 1):
            all_data = process_week(week)
        
            if all_data:
                print(f"Processed {len(all_data)} rows of data for Week {week}")
                # Here you can add code to save the data to a file or database
            else:
                print("No data was processed.")


# insert new teams
# insert new players
# insert stats
# insert plays