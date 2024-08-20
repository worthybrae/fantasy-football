import os
import time
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, WebDriverException, StaleElementReferenceException

class Browser:
    def __init__(self, headless=True, max_retries=3, retry_delay=.25):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.driver = self._initialize_driver(headless)

    def _initialize_driver(self, headless):
        current_dir = os.getcwd()
        chromedriver_path = os.path.join(current_dir, "chromedriver")
        if os.name == 'nt':  # For Windows
            chromedriver_path += ".exe"

        if not os.path.exists(chromedriver_path):
            self._run_chromedriver_init_script()

        service = Service(chromedriver_path)
        chrome_options = webdriver.ChromeOptions()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")

        return webdriver.Chrome(service=service, options=chrome_options)

    def _run_chromedriver_init_script(self):
        script_path = os.path.join(os.getcwd(), "initialize_chromedriver.py")
        if os.path.exists(script_path):
            subprocess.run(["python", script_path], check=True)
        else:
            raise FileNotFoundError("initialize_chromedriver.py script not found in the root directory.")

    def _retry(self, func, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (TimeoutException, WebDriverException, StaleElementReferenceException) as e:
                print(f"Error during request: {e}")
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(self.retry_delay)

    def get_page(self, url):
        def _get():
            self.driver.get(url)
            WebDriverWait(self.driver, 3).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        self._retry(_get)

    def scroll_to_element(self, element_type, element_id, max_scrolls=100, scroll_pause_time=0.1):
        def is_element_in_viewport():
            return self.driver.execute_script(f"""
                var elem = document.{element_type}('{element_id}');
                if (!elem) return false;
                var rect = elem.getBoundingClientRect();
                return (
                    rect.top >= 0 &&
                    rect.left >= 0 &&
                    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
                    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
                );
            """)

        def _scroll():
            scrolls = 0
            while not is_element_in_viewport() and scrolls < max_scrolls:
                self.driver.execute_script("window.scrollBy(0, 500);")
                time.sleep(scroll_pause_time)
                scrolls += 1

                try:
                    element = self.driver.find_element(By.ID, element_id)
                    if element.is_displayed():
                        ActionChains(self.driver).move_to_element(element).perform()
                        return element
                except:
                    pass  # Element not found yet, continue scrolling

            # If we've exited the loop without finding the element, try one last time
            element = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.ID, element_id))
            )
            ActionChains(self.driver).move_to_element(element).perform()
            return element

        return self._retry(_scroll)

    def get_page_source(self):
        return self.driver.page_source

    def close(self):
        self.driver.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()