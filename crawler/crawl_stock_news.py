import requests
from bs4 import BeautifulSoup
import json
import re
import time
from urllib.parse import urljoin
import logging
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cafef_crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_cafef_article(url):
    """Extract article data (title, date, content) from a CafeF article URL"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching article URL {url}: {e}")
        return None
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract the title
    title_element = soup.find('h1', class_='title')
    title = title_element.get_text(strip=True) if title_element else "Title not found"
    
    # Extract the date - try multiple approaches
    date = "Date not found"
    
    # Approach 1: Look for span with class pdate
    date_element = soup.find('span', class_='pdate')
    if date_element:
        date_text = date_element.get_text(strip=True)
        # Extract only the date part (DD-MM-YYYY)
        date_match = re.search(r'(\d{2}-\d{2}-\d{4})', date_text)
        if date_match:
            date = date_match.group(1)
    
    # If date still not found, try alternative approaches
    if date == "Date not found":
        # Look for other common date patterns in the HTML
        text = soup.get_text()
        date_patterns = [
            r'(\d{2}-\d{2}-\d{4})',  # DD-MM-YYYY
            r'(\d{2}/\d{2}/\d{4})'   # DD/MM/YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            if matches:
                date = matches[0]
                break
    
    # Extract the content
    content_elements = []
    
    # First, try to find the sapo (introduction/summary)
    sapo_element = soup.find('h2', class_='sapo')
    if sapo_element:
        sapo_text = sapo_element.get_text(strip=True)
        sapo_text = re.sub(r'\s+', ' ', sapo_text).strip('" ')
        if sapo_text:
            content_elements.append(sapo_text)
    
    # Try different possible selectors for the main content
    content_selectors = [
        ('div', {'class': 'detail-content'}),
        ('div', {'data-role': 'content'}),
        ('div', {'class': 'detail-cmain'}),
        ('div', {'class': 'contentdetail'})
    ]
    
    content_container = None
    for tag, attrs in content_selectors:
        content_container = soup.find(tag, attrs)
        if content_container:
            break
    
    if content_container:
        # Get all paragraphs from the content
        paragraphs = content_container.find_all('p')
        
        for paragraph in paragraphs:
            # Skip empty paragraphs or those only containing ellipses
            p_text = paragraph.get_text(strip=True)
            if p_text and not p_text == '...' and p_text != '.' and len(p_text) > 1:
                # Clean up the text
                p_text = re.sub(r'\s+', ' ', p_text)
                # Remove quotation marks at beginning and end
                p_text = p_text.strip('" ')
                if p_text:
                    content_elements.append(p_text)
    
    # Join all content elements with newlines in between
    content = '\n\n'.join(content_elements)
    
    # Create a dictionary with the extracted data
    article_data = {
        "url": url,
        "title": title,
        "date": date,
        "content": content
    }
    
    return article_data

def crawl_cafef_links(url, max_links=20):
    """Crawl article links from a CafeF category page"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching category URL {url}: {e}")
        return []
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find article links - CafeF typically has news links in <a> tags with href attributes
    # containing specific patterns
    article_links = []
    
    # Common link containers on CafeF
    link_containers = [
        soup.find_all('a', class_='avatar'),  # Main news
        soup.find_all('a', class_='title-news'),  # News titles
        soup.find_all('a', class_='link-content-news'),  # Content links
        soup.find_all('h3', class_='title') # Title containers
    ]
    
    # Process all link containers
    for container in link_containers:
        for item in container:
            # For h3 tags, look for the 'a' tag inside
            if item.name == 'h3':
                link_tag = item.find('a')
                if not link_tag:
                    continue
            else:
                link_tag = item
            
            href = link_tag.get('href')
            if href and '.chn' in href:  # CafeF articles typically end with .chn
                full_url = urljoin('https://cafef.vn', href)
                if full_url not in article_links:
                    article_links.append(full_url)
    
    # If we still don't have many links, look for all 'a' tags with '.chn' in href
    if len(article_links) < max_links:
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if '.chn' in href and '//' not in href:  # Avoid external links
                full_url = urljoin('https://cafef.vn', href)
                if full_url not in article_links:
                    article_links.append(full_url)
    
    # Limit the number of links to process
    return article_links[:max_links]

def crawl_and_extract_cafef(category_url, max_links=20, output_file="cafef_articles.json"):
    """Crawl links from a category page and extract article content from each"""
    # Get article links
    logger.info(f"Crawling links from: {category_url}")
    article_links = crawl_cafef_links(category_url, max_links)
    logger.info(f"Found {len(article_links)} article links")
    
    # Extract articles
    articles = []
    for i, link in enumerate(article_links):
        logger.info(f"Processing article {i+1}/{len(article_links)}: {link}")
        
        article_data = extract_cafef_article(link)
        if article_data:
            articles.append(article_data)
            logger.info(f"Successfully extracted article: {article_data['title']}")
        else:
            logger.warning(f"Failed to extract article from: {link}")
        
        # Polite delay to avoid overloading the server
        time.sleep(random.uniform(1, 3))
    
    # Save results to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved {len(articles)} articles to {output_file}")
    except Exception as e:
        logger.error(f"Error saving to JSON: {e}")
    
    return articles

if __name__ == "__main__":
    # The URL of the CafeF stock market news page
    category_url = "https://cafef.vn/thi-truong-chung-khoan.chn"
    
    # Number of articles to extract
    max_articles = 1000  # You can adjust this number
    
    # Output file
    output_file = "cafef_stock_articles.json"
    
    # Run the crawling and extraction
    articles = crawl_and_extract_cafef(category_url, max_articles, output_file)
    
    # Print summary
    if articles:
        print(f"\nSuccessfully crawled and extracted {len(articles)} articles.")
        print(f"Results saved to: {output_file}")
    else:
        print("\nNo articles were successfully extracted.")