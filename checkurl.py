import pandas as pd
import requests
from tqdm import tqdm

# Load the CSV file
df = pd.read_csv(r'C:\Users\Desktop\Documents\index_check.csv')

# Function to check if a URL exists
def check_url_existence(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return 1 if response.status_code == 200 else 0
    except requests.RequestException:
        return 0

# Apply the function to each URL in the 'G-URL' column with a progress bar, one by one
tqdm.pandas(desc="Checking URLs")
df['Aval'] = df['G-URL'].progress_apply(check_url_existence)

# Save the updated DataFrame to a new CSV file
df.to_csv(r'C:\Users\Desktop\Documents\index_check_updated2.csv', index=False)

print("The 'Aval' column has been updated based on the existence of URLs.")
