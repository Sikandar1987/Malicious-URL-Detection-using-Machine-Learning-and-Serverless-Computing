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


# Function to extract webpage content features
def extract_content_features(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        features = {}
        # Basic content features
        features['num_images'] = len(soup.find_all('img'))
        features['num_links'] = len(soup.find_all('a'))
        features['num_words'] = len(soup.get_text().split())
        features['num_headings'] = len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
        features['num_meta'] = len(soup.find_all('meta'))
        
        # Meta tag analysis
        description_tag = soup.find('meta', {'name': 'description'})
        features['has_meta_description'] = 1 if description_tag else 0
        features['meta_description_length'] = len(description_tag['content']) if description_tag and 'content' in description_tag.attrs else 0
        
        keywords_tag = soup.find('meta', {'name': 'keywords'})
        features['has_meta_keywords'] = 1 if keywords_tag else 0
        features['meta_keywords_length'] = len(keywords_tag['content']) if keywords_tag and 'content' in keywords_tag.attrs else 0
        
        # Advanced content analysis
        title = soup.find('title').get_text() if soup.find('title') else ''
        features['title_length'] = len(title)
        features['num_scripts'] = len(soup.find_all('script'))
        features['num_styles'] = len(soup.find_all('style'))
        
        # Counting specific HTML elements
        features['num_iframes'] = len(soup.find_all('iframe'))
        features['num_buttons'] = len(soup.find_all('button'))
        features['num_forms'] = len(soup.find_all('form'))
        features['num_inputs'] = len(soup.find_all('input'))
        
        # Textual content analysis
        text = soup.get_text()
        features['avg_word_length'] = sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
        features['num_uppercase_words'] = sum(1 for word in text.split() if word.isupper())
        features['num_digits_in_text'] = sum(c.isdigit() for c in text)
        features['num_special_chars'] = sum(not c.isalnum() for c in text)
        
        # New content features
        features['has_video'] = 1 if soup.find('video') else 0
        features['num_bold'] = len(soup.find_all('b')) + len(soup.find_all('strong'))
        features['num_italic'] = len(soup.find_all('i')) + len(soup.find_all('em'))
        features['num_tables'] = len(soup.find_all('table'))
        features['num_lists'] = len(soup.find_all(['ul', 'ol']))
        features['has_canonical_tag'] = 1 if soup.find('link', {'rel': 'canonical'}) else 0
        features['has_favicon'] = 1 if soup.find('link', {'rel': 'icon'}) else 0
        features['num_embeds'] = len(soup.find_all(['embed', 'object']))
        features['has_open_graph'] = 1 if soup.find('meta', {'property': 'og:title'}) else 0
        
        return features
    except requests.RequestException:
        return {'num_images': 0, 'num_links': 0, 'num_words': 0, 'num_headings': 0, 'num_meta': 0, 'has_meta_description': 0,
                'meta_description_length': 0, 'has_meta_keywords': 0, 'meta_keywords_length': 0, 'title_length': 0, 'num_scripts': 0,
                'num_styles': 0, 'num_iframes': 0, 'num_buttons': 0, 'num_forms': 0, 'num_inputs': 0, 'avg_word_length': 0,
                'num_uppercase_words': 0, 'num_digits_in_text': 0, 'num_special_chars': 0, 'has_video': 0, 'num_bold': 0,
                'num_italic': 0, 'num_tables': 0, 'num_lists': 0, 'has_canonical_tag': 0, 'has_favicon': 0, 'num_embeds': 0, 'has_open_graph': 0}

# Function to process each URL and extract all features
def process_url(url):
    features = {}
    
      
    # Content-based features
    content_features = extract_content_features(url)
    features.update(content_features)
    
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
    df.to_csv(r'C:\urls_with_content-based_features.csv', index=False)

print("Features extracted and saved to 'urls_with_contetn-based_features.csv'.")
