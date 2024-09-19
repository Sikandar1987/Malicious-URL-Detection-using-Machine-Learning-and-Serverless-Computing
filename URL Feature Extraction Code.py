import pandas as pd
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import concurrent.futures

# Load the CSV file
df = pd.read_csv(r'C:\urls.csv')

# Function to check if a URL is live
def is_live(url):
    try:
        response = requests.get(url, timeout=5)
        return 1 if response.status_code == 200 else 0
    except requests.RequestException:
        return 0

# Function to extract URL string-based features
def extract_url_features(url):
    parsed_url = urlparse(url)
    features = {}
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['num_slashes'] = url.count('/')
    features['num_params'] = len(parsed_url.query.split('&')) if parsed_url.query else 0
    features['has_https'] = 1 if parsed_url.scheme == 'https' else 0
    features['has_www'] = 1 if 'www' in parsed_url.netloc else 0
    features['domain_length'] = len(parsed_url.netloc)
    features['path_length'] = len(parsed_url.path)
    features['subdomain_length'] = len(parsed_url.hostname.split('.')[0]) if len(parsed_url.hostname.split('.')) > 2 else 0
    features['is_ip'] = 1 if re.match(r'[0-9]+(?:\.[0-9]+){3}', parsed_url.netloc) else 0
    features['num_digits'] = sum(c.isdigit() for c in url)
    features['has_query'] = 1 if parsed_url.query else 0
    features['num_special_chars_in_url'] = sum(not c.isalnum() for c in url)
    features['num_uppercase_in_url'] = sum(1 for c in url if c.isupper())
    features['is_shortened_url'] = 1 if 'bit.ly' in url or 't.co' in url else 0
    features['url_entropy'] = len(set(url)) / len(url)
    features['top_level_domain'] = parsed_url.netloc.split('.')[-1]
    features['url_depth'] = len(parsed_url.path.strip('/').split('/')) if parsed_url.path.strip('/') else 0
    features['has_at_symbol'] = 1 if '@' in url else 0
    features['has_hash_symbol'] = 1 if '#' in url else 0
    features['has_subdomain'] = 1 if len(parsed_url.hostname.split('.')) > 2 else 0
    features['has_port'] = 1 if parsed_url.port else 0
    features['num_alphabets_in_url'] = sum(c.isalpha() for c in url)
    features['is_url_encoded'] = 1 if '%' in url else 0
    features['is_long_url'] = 1 if len(url) > 75 else 0
    return features

# Function to process each URL and extract all features
def process_url(url):
    features = {}
    
    # URL-based features
    url_features = extract_url_features(url)
    features.update(url_features)
        
    # Live status
    features['live'] = is_live(url)
    
    return features

# Multi-threading for faster processing with progress bar and saving after every 1000 URLs
batch_size = 1000
for batch_start in range(0, len(df), batch_size):
    batch_end = min(batch_start + batch_size, len(df))
    batch_df = df.iloc[batch_start:batch_end]

    # Multi-threading for faster processing with progress bar
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_url, batch_df['url']), total=len(batch_df)))

    # Append results to the dataframe
    for i, result in enumerate(results):
        for key, value in result.items():
            df.loc[batch_start + i, key] = value

    # Save after every 1000 URLs
    df.to_csv(r'C:\urls_With_URL_Features.csv', index=False)

print("Features extracted and saved to 'urls_with_features.csv'.")
