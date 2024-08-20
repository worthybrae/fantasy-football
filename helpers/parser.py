from bs4 import BeautifulSoup


def process_table(page_source, table_id, columns, indexes):
    # Convert page source to BeautifulSoup object
    soup = BeautifulSoup(page_source, 'html.parser')
    # Find the table with the given id
    table = soup.find('table', id=table_id)
    # If the table was found, return its data
    if table:
        # Find all rows in the table body
        rows = table.find('tbody').find_all('tr')
        
        data = []
        # Iterate through each row
        for row in rows:
            row_data = []
            for header in row.find_all('th'):
                row_data.append(header.text.strip())
            for cell in row.find_all('td'):
                row_data.append(cell.text.strip())
            
            data.append(row_data)
        return data
    else:
        print(f"Table with id '{table_id}' not found.")
        return None

