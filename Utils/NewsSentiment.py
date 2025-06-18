import json
import pandas as pd
import requests
import time
from datetime import datetime
import os

# Multiple API keys for rotation
GEMINI_API_KEYS = [
    "AIzaSyBKwU2iZ6I5Szys6pVpYU-AtZP1q0ok8JM",
    "AIzaSyDB_rWy43YUj21cdse7Xd0B50qOkIfNXKg",
    "AIzaSyB-yf6GoVc_cKtMUUTl8DIIabVkCmoxYQA",
    "AIzaSyBbWR_oV_55_uS0jzGxwmeIhXSwrvldUss",
    "AIzaSyBvyDPjPBj11dQjhWj9WIEMGeE7Z9zXLuQ",
    "AIzaSyAyRtmQSHS1GrRahV0xO312ox6WVom_fLA"
]

current_key_index = 0

# File to store processed URLs
PROCESSED_URLS_FILE = 'data/processed_urls.json'

def load_processed_urls():
    """Load the set of already processed URLs"""
    if os.path.exists(PROCESSED_URLS_FILE):
        with open(PROCESSED_URLS_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_processed_urls(processed_urls):
    """Save the set of processed URLs"""
    os.makedirs(os.path.dirname(PROCESSED_URLS_FILE), exist_ok=True)
    with open(PROCESSED_URLS_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(processed_urls), f)

def get_next_api_key():
    """Rotate API keys"""
    global current_key_index
    key = GEMINI_API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
    return key

def gemini_sentiment(text, max_retries=3):
    """Get sentiment from Gemini API with key rotation"""
    prompt = f"""
    Phân tích cảm xúc tin tức chứng khoán Việt Nam này. Trả về JSON:
    {{"sentiment_score": <-1 đến 1>, "confidence": <0 đến 1>}}
    
    Tin: {text[:800]}
    """
    
    used_keys = set()  # Track which keys have been used for this request
    
    while len(used_keys) < len(GEMINI_API_KEYS):  # Try all available keys
        api_key = get_next_api_key()
        if api_key in used_keys:
            continue
            
        used_keys.add(api_key)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"  
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 100,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['candidates'][0]['content']['parts'][0]['text']
                
                # Extract JSON
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return data.get('sentiment_score', 0.0)
            
            elif response.status_code == 429:  # Rate limit
                print(f"Rate limit hit with key {api_key[-4:]}, trying next key...")
                time.sleep(2)
                continue
                
        except Exception as e:
            print(f"API error with key {api_key[-4:]}: {e}")
            time.sleep(1)
            continue
    
    # If we've tried all keys and none worked, raise an exception
    raise Exception("All API keys have been exhausted. Please try again later.")

def save_detailed_results(new_result):
    """Save detailed sentiment analysis results for each article"""
    detailed_file = 'data/detailed_sentiment_results.json'
    
    # Load existing results if any
    if os.path.exists(detailed_file):
        with open(detailed_file, 'r', encoding='utf-8') as f:
            detailed_results = json.load(f)
    else:
        detailed_results = []
    
    # Add timestamp
    new_result['processed_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    detailed_results.append(new_result)
    
    # Save updated results
    os.makedirs(os.path.dirname(detailed_file), exist_ok=True)
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

def analyze_news(news_data):
    """Analyze sentiment for all news"""
    results = []
    processed_urls = load_processed_urls()
    new_urls = set()
    
    # Load existing results if any
    results_file = 'data/sentiment_results.csv'
    if os.path.exists(results_file):
        results = pd.read_csv(results_file).to_dict('records')
        print(f"Loaded {len(results)} existing results")
    
    for i, news in enumerate(news_data):
        url = news.get('url', '')
        if url in processed_urls:
            print(f"Skipping already processed article {i+1}/{len(news_data)}")
            continue
            
        print(f"Processing {i+1}/{len(news_data)}")
        
        title = news.get('title', '')
        content = news.get('content', '')
        full_text = f"{title}. {content}"
        
        try:
            # Calculate sentiment
            gemini_score = gemini_sentiment(full_text)
            final_score = gemini_score
            
            new_result = {
                'date': news.get('date', ''),
                'title': title,
                'url': url,
                'source': news.get('source', ''),
                'gemini_sentiment': gemini_score,
                'final_sentiment': final_score,
                'content_preview': content[:200] + '...' if len(content) > 200 else content
            }
            
            results.append(new_result)
            new_urls.add(url)
            
            # Save detailed results
            save_detailed_results(new_result)
            
            # Save results after each article
            df = pd.DataFrame(results)
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            df.to_csv(results_file, index=False, encoding='utf-8')
            
            # Update processed URLs
            processed_urls.update(new_urls)
            save_processed_urls(processed_urls)
            
        except Exception as e:
            print(f"Error processing article {i+1}: {e}")
            print("Stopping processing due to API key exhaustion")
            break
            
        time.sleep(0.5)  # Rate limiting
    
    return pd.DataFrame(results)

def merge_with_stock_data(sentiment_df, stock_csv_path):
    """Merge sentiment with stock data"""
    # Load stock data
    stock_df = pd.read_csv(stock_csv_path)
    
    # Convert dates
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], format='%d-%m-%Y', errors='coerce')
    stock_df['data_date'] = pd.to_datetime(stock_df['data_date'], errors='coerce')
    
    # Aggregate sentiment by date
    daily_sentiment = sentiment_df.groupby('date').agg({
        'gemini_sentiment': 'mean', 
        'final_sentiment': 'mean',
        'title': 'count'
    }).rename(columns={'title': 'news_count'}).reset_index()
    
    # Merge
    merged = stock_df.merge(daily_sentiment, left_on='data_date', right_on='date', how='left')
    merged['final_sentiment'] = merged['final_sentiment'].fillna(0)
    merged['news_count'] = merged['news_count'].fillna(0)
    
    return merged

def map_sentiment_to_stock():
    """Map sentiment results to stock data"""
    print("Mapping sentiment to stock data...")
    
    # Load sentiment results
    sentiment_file = 'data/sentiment_results.csv'
    if not os.path.exists(sentiment_file):
        print("No sentiment results found")
        return None
        
    sentiment_df = pd.read_csv(sentiment_file)
    
    # Merge with stock data
    merged_df = merge_with_stock_data(sentiment_df, 'data/VN100_stock_price_1D.csv')
    
    # Save mapping results
    mapping_file = 'data/stock_sentiment_mapping.csv'
    merged_df.to_csv(mapping_file, index=False, encoding='utf-8')
    print(f"Mapping results saved to {mapping_file}")
    
    # Summary
    avg_sentiment = merged_df['final_sentiment'].mean()
    positive_days = (merged_df['final_sentiment'] > 0.1).sum()
    negative_days = (merged_df['final_sentiment'] < -0.1).sum()
    
    print(f"\nMapping Results:")
    print(f"Total days mapped: {len(merged_df)}")
    print(f"Average sentiment: {avg_sentiment:.3f}")
    print(f"Positive days: {positive_days}")
    print(f"Negative days: {negative_days}")
    
    return merged_df

def main():
    """Main function"""
    # Load news data
    with open('cafef_stock_articles.json', 'r', encoding='utf-8') as f:
        news_data = json.load(f)
    
    print(f"Found {len(news_data)} news articles...")
    
    # Analyze sentiment
    sentiment_df = analyze_news(news_data)
    
    if not sentiment_df.empty:
        print("Sentiment analysis saved to sentiment_results.csv")
        print("Detailed results saved to detailed_sentiment_results.json")
        
        # Map sentiment to stock data
        map_sentiment_to_stock()
        
        return sentiment_df
    else:
        print("No new articles to process")
        return None

if __name__ == "__main__":
    main()