from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

def test_selenium_with_chromedriver():
    # Get the current directory
    current_dir = os.getcwd()

    # Specify the path to the ChromeDriver executable
    chromedriver_path = os.path.join(current_dir, "chromedriver")
    if os.name == 'nt':  # For Windows
        chromedriver_path += ".exe"

    # Set up the Chrome service with the specified ChromeDriver path
    service = Service(chromedriver_path)

    # Set up Chrome options (optional)
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--start-maximized")  # Start the browser maximized

    # Initialize the Chrome driver
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Navigate to a website
        driver.get("https://www.python.org")

        # Wait for the search input to be present
        search_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )

        # Perform a search
        search_input.send_keys("Selenium")
        search_input.submit()

        # Wait for the results
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "h3.event-widget-title"))
        )

        # Print the title of the page
        print(f"Page title: {driver.title}")

        # Find and print the first search result
        first_result = driver.find_element(By.CSS_SELECTOR, "#content > div > section > form > ul > li")
        print(f"First search result: {first_result.text}")

        print("Selenium test completed successfully!")

    except Exception as e:
        print(f"An error occurred during the Selenium test: {e}")

    finally:
        # Close the browser
        driver.quit()

if __name__ == "__main__":
    test_selenium_with_chromedriver()