import os
import sys
import subprocess
import requests
import zipfile
import shutil

def get_chrome_version():
    if sys.platform.startswith('darwin'):  # macOS
        try:
            cmd = ['osascript', '-e', 'tell application "Google Chrome" to version']
            chrome_version = subprocess.check_output(cmd).decode('utf-8').strip()
        except subprocess.CalledProcessError:
            chrome_paths = [
                '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                '/Applications/Google Chrome.app/Contents/Versions/*/Google Chrome',
            ]
            for path in chrome_paths:
                try:
                    cmd = [path, '--version']
                    output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode('utf-8')
                    chrome_version = output.strip().split()[-1]
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            else:
                raise Exception("Could not determine Chrome version. Is Chrome installed?")
    elif sys.platform.startswith('win'):
        cmd = r'reg query "HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon" /v version'
        output = subprocess.check_output(cmd).decode('utf-8')
        chrome_version = output.strip().split()[-1]
    elif sys.platform.startswith('linux'):
        cmd = 'google-chrome --version'
        output = subprocess.check_output(cmd, shell=True).decode('utf-8')
        chrome_version = output.strip().split()[-1]
    else:
        raise OSError("Unsupported operating system")
    return chrome_version

def get_latest_stable_version():
    url = "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions.json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data['channels']['Stable']['version']

def download_chromedriver(version):
    base_url = "https://storage.googleapis.com/chrome-for-testing-public"
    
    if sys.platform.startswith('win'):
        platform = "win32"
    elif sys.platform.startswith('darwin'):
        platform = "mac-arm64" if "arm" in os.uname().machine.lower() else "mac-x64"
    elif sys.platform.startswith('linux'):
        platform = "linux64"
    else:
        raise OSError("Unsupported operating system")

    download_url = f"{base_url}/{version}/{platform}/chromedriver-{platform}.zip"

    try:
        response = requests.get(download_url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error downloading ChromeDriver: {e}")
        return

    zip_file_path = "chromedriver.zip"
    with open(zip_file_path, "wb") as zip_file:
        zip_file.write(response.content)

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract to a temporary directory
            temp_dir = "temp_extract"
            zip_ref.extractall(temp_dir)
        
        # Move only the chromedriver executable
        chromedriver_name = "chromedriver.exe" if sys.platform.startswith('win') else "chromedriver"
        src_path = os.path.join(temp_dir, f"chromedriver-{platform}", chromedriver_name)
        dst_path = os.path.join(os.getcwd(), chromedriver_name)
        
        # Remove existing chromedriver if it exists
        if os.path.exists(dst_path):
            os.remove(dst_path)
        
        shutil.move(src_path, dst_path)
        
        # Clean up
        shutil.rmtree(temp_dir)
        os.remove(zip_file_path)
        
        # Make chromedriver executable on Unix-like systems
        if not sys.platform.startswith('win'):
            os.chmod(dst_path, 0o755)
        
    except Exception as e:
        print(f"Error extracting ChromeDriver: {e}")
        return

    print(f"ChromeDriver {version} has been downloaded and placed in the current directory: {dst_path}")

def main():
    try:
        chrome_version = get_chrome_version()
        print(f"Detected Chrome version: {chrome_version}")
        
        latest_stable_version = get_latest_stable_version()
        print(f"Latest stable ChromeDriver version: {latest_stable_version}")
        
        download_chromedriver(latest_stable_version)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()