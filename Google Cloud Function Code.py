from flask import Flask, request, render_template
import pandas as pd
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
import pickle
import os
import time  # For calculating execution time
from tqdm import tqdm  # Progress bar

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to check if a URL is live
def is_live(url):
    try:
        response = requests.get(url, timeout=5)
        return 1 if response.status_code == 200 else 0
    except requests.RequestException:
        return 0

# Function to extract URL-based features
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
    features['url_depth'] = len(parsed_url.path.strip('/').split('/')) if parsed_url.path.strip('/') else 0
    features['has_at_symbol'] = 1 if '@' in url else 0
    features['has_hash_symbol'] = 1 if '#' in url else 0
    features['has_subdomain'] = 1 if len(parsed_url.hostname.split('.')) > 2 else 0
    features['has_port'] = 1 if parsed_url.port else 0
    features['num_alphabets_in_url'] = sum(c.isalpha() for c in url)
    features['is_url_encoded'] = 1 if '%' in url else 0
    features['is_long_url'] = 1 if len(url) > 75 else 0
    return features

# Function to extract webpage content-based features
def extract_content_features(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')

        features = {}
        features['num_images'] = len(soup.find_all('img'))
        features['num_links'] = len(soup.find_all('a'))
        features['num_words'] = len(soup.get_text().split())
        features['num_headings'] = len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
        features['num_meta'] = len(soup.find_all('meta'))

        description_tag = soup.find('meta', {'name': 'description'})
        features['has_meta_description'] = 1 if description_tag else 0
        features['meta_description_length'] = len(description_tag['content']) if description_tag and 'content' in description_tag.attrs else 0

        keywords_tag = soup.find('meta', {'name': 'keywords'})
        features['has_meta_keywords'] = 1 if keywords_tag else 0
        features['meta_keywords_length'] = len(keywords_tag['content']) if keywords_tag and 'content' in keywords_tag.attrs else 0

        title = soup.find('title').get_text() if soup.find('title') else ''
        features['title_length'] = len(title)
        features['num_scripts'] = len(soup.find_all('script'))
        features['num_styles'] = len(soup.find_all('style'))

        features['num_iframes'] = len(soup.find_all('iframe'))
        features['num_buttons'] = len(soup.find_all('button'))
        features['num_forms'] = len(soup.find_all('form'))
        features['num_inputs'] = len(soup.find_all('input'))

        text = soup.get_text()
        features['avg_word_length'] = sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
        features['num_uppercase_words'] = sum(1 for word in text.split() if word.isupper())
        features['num_digits_in_text'] = sum(c.isdigit() for c in text)
        features['num_special_chars'] = sum(not c.isalnum() for c in text)

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
        # Return default values for all features if the request fails
        return {
            'num_images': 0,
            'num_links': 0,
            'num_words': 0,
            'num_headings': 0,
            'num_meta': 0,
            'has_meta_description': 0,
            'meta_description_length': 0,
            'has_meta_keywords': 0,
            'meta_keywords_length': 0,
            'title_length': 0,
            'num_scripts': 0,
            'num_styles': 0,
            'num_iframes': 0,
            'num_buttons': 0,
            'num_forms': 0,
            'num_inputs': 0,
            'avg_word_length': 0,
            'num_uppercase_words': 0,
            'num_digits_in_text': 0,
            'num_special_chars': 0,
            'has_video': 0,
            'num_bold': 0,
            'num_italic': 0,
            'num_tables': 0,
            'num_lists': 0,
            'has_canonical_tag': 0,
            'has_favicon': 0,
            'num_embeds': 0,
            'has_open_graph': 0
        }

# Function to predict with all models
def predict_with_all_models(url):
    # Start feature extraction timer
    feature_extraction_start = time.time()
    
    url_features = extract_url_features(url)
    content_features = extract_content_features(url)
    
    feature_extraction_time = time.time() - feature_extraction_start

    all_features = {}
    all_features.update(url_features)
    all_features.update(content_features)

    url_feature_df = pd.DataFrame([url_features])
    content_feature_df = pd.DataFrame([content_features])
    all_feature_df = pd.DataFrame([all_features])

    # Start algorithm running timer
    algo_running_start = time.time()

    with open('random_forest_model.pkl', 'rb') as f:
        main_model = pickle.load(f)
    main_prediction = main_model.predict(all_feature_df)

    with open('random_forest_model_URL.pkl', 'rb') as f:
        url_model = pickle.load(f)
    url_prediction = url_model.predict(url_feature_df)

    with open('random_forest_model_Content.pkl', 'rb') as f:
        content_model = pickle.load(f)
    content_prediction = content_model.predict(content_feature_df)

    algo_running_time = time.time() - algo_running_start

    return {
        'main_result': 'Phishing' if main_prediction[0] == 1 else 'Not Phishing',
        'url_result': 'Phishing' if url_prediction[0] == 1 else 'Not Phishing',
        'content_result': 'Phishing' if content_prediction[0] == 1 else 'Not Phishing',
        'feature_extraction_time': feature_extraction_time,
        'algo_running_time': algo_running_time
    }
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the file is uploaded
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and file.filename.endswith('.csv'):
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Start total execution timer
            total_start_time = time.time()

            # Read CSV and extract URLs
            df = pd.read_csv(file_path)
            if 'url' not in df.columns:
                return "CSV must contain 'url' column", 400

            # Accumulate feature extraction and algorithm running times
            total_feature_extraction_time = 0
            total_algo_running_time = 0

            # Process each URL and get predictions with a progress bar in the terminal
            results = []
            for url in tqdm(df['url'], desc="Processing URLs", unit="url"):
                prediction = predict_with_all_models(url)
                total_feature_extraction_time += prediction['feature_extraction_time']
                total_algo_running_time += prediction['algo_running_time']
                results.append({'url': url, 'result': prediction})

            # Stop the total execution timer
            total_time = time.time() - total_start_time

            # Render the results and display the total times
            return render_template('results.html', results=results, total_time=total_time,
                                   feature_extraction_time=total_feature_extraction_time,
                                   algo_running_time=total_algo_running_time)

    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
