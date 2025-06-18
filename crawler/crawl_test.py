import requests
from bs4 import BeautifulSoup
import json
import re
import time
from urllib.parse import urljoin, urlparse
import logging
import random

# Set up logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cafef_crawler.log", encoding='utf-8'),
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

def extract_category_id(url):
    """Extract the category ID from the URL for API pagination"""
    parsed_url = urlparse(url)
    path = parsed_url.path
    
    # Try to find a pattern like /category-name/ID.chn
    match = re.search(r'/([^/]+)/(\d+)\.chn', path)
    if match:
        return match.group(2)
    
    # If the above pattern doesn't match, try more general patterns
    # Look for any number followed by .chn
    match = re.search(r'(\d+)\.chn', path)
    if match:
        return match.group(1)
    
    # If we can't find a category ID, return a default value
    return "18831"  # Default ID for stock market category

def extract_links_from_html(html_content, base_url):
    """Extract article links from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    article_links = []
    
    # Common link containers on CafeF
    link_containers = [
        soup.find_all('a', class_='avatar'),  # Main news
        soup.find_all('a', class_='title-news'),  # News titles
        soup.find_all('a', class_='link-content-news'),  # Content links
        soup.find_all('h3', class_='title'), # Title containers
        soup.find_all('h2', class_='title'),  # More title containers
        soup.find_all('h4', class_='title')   # More title containers
    ]
    
    # Process all link containers
    for container in link_containers:
        for item in container:
            # For h3/h2/h4 tags, look for the 'a' tag inside
            if item.name in ['h2', 'h3', 'h4']:
                link_tag = item.find('a')
                if not link_tag:
                    continue
            else:
                link_tag = item
            
            href = link_tag.get('href')
            if href and '.chn' in href:  # CafeF articles typically end with .chn
                full_url = urljoin(base_url, href)
                if full_url not in article_links:
                    article_links.append(full_url)
    
    # If we don't have many links, look for all 'a' tags with '.chn' in href
    if len(article_links) < 10:
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if '.chn' in href and '//' not in href:  # Avoid external links
                full_url = urljoin(base_url, href)
                if full_url not in article_links:
                    article_links.append(full_url)
    
    return article_links

def extract_links_from_api_response(api_response, base_url):
    """Extract article links from API response (which appears to be HTML directly)"""
    article_links = []
    
    try:
        # Based on the example provided, the API returns HTML directly
        soup = BeautifulSoup(api_response, 'html.parser')
        
        # Find all article container divs with the specific class and data-id
        article_containers = soup.find_all('div', class_='tlitem box-category-item')
        
        if article_containers:
            logger.info(f"Found {len(article_containers)} article containers in API response")
            
            for container in article_containers:
                # Look for links inside the containers
                links = container.find_all('a', href=True)
                
                for link in links:
                    href = link.get('href')
                    if href and '.chn' in href:  # CafeF articles typically end with .chn
                        # Handle both absolute and relative URLs
                        if href.startswith('http'):
                            full_url = href
                        else:
                            full_url = urljoin(base_url, href)
                            
                        if full_url not in article_links:
                            article_links.append(full_url)
                            # For debugging
                            logger.debug(f"Found article link: {full_url}")
        else:
            # If we don't find the specific structure, fall back to a more general approach
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href and '.chn' in href:  # CafeF articles typically end with .chn
                    full_url = urljoin(base_url, href)
                    if full_url not in article_links:
                        article_links.append(full_url)
    
    except Exception as e:
        logger.error(f"Error parsing API response: {e}")
        # Print a small sample of the response for debugging
        sample = api_response[:500] + "..." if len(api_response) > 500 else api_response
        logger.error(f"Response sample: {sample}")
    
    return article_links

def crawl_cafef_links(url, max_links=100, pages=10):
    """Crawl article links from CafeF using API pagination for subsequent pages
    
    Args:
        url: Base URL of the category page
        max_links: Maximum number of links to retrieve
        pages: Number of pages to crawl
    
    Returns:
        List of article URLs
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Extract the category ID for API pagination
    category_id = extract_category_id(url)
    logger.info(f"Extracted category ID: {category_id}")
    
    article_links = []
    
    # First page: Crawl the main URL
    logger.info(f"Crawling first page: {url}")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        first_page_links = extract_links_from_html(response.text, "https://cafef.vn")
        
        # Add links to the result list
        for link in first_page_links:
            if link not in article_links:
                article_links.append(link)
        
        logger.info(f"Found {len(first_page_links)} links on first page")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching first page {url}: {e}")
    
    # If we already have enough links or there's only one page, return early
    if len(article_links) >= max_links or pages <= 1:
        return article_links[:max_links]
    
    # Subsequent pages: Use the API
    for page in range(2, pages + 1):
        # CafeF uses a mobile API for pagination based on the screenshot
        api_url = f"https://m.cafef.vn/timelinelist/{category_id}/{page}.chn"
        logger.info(f"Crawling page {page} via API: {api_url}")
        
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            
            # Based on the screenshot, the API returns HTML directly
            page_links = extract_links_from_api_response(response.text, "https://cafef.vn")
            
            # Add new links to the result list
            new_links_count = 0
            for link in page_links:
                if link not in article_links:
                    article_links.append(link)
                    new_links_count += 1
            
            logger.info(f"Found {new_links_count} new links on page {page}")
            
            # If we didn't find any new links, or we have enough links, stop crawling
            if new_links_count == 0 or len(article_links) >= max_links:
                break
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching page {page} via API {api_url}: {e}")
        
        # Polite delay between API requests
        if page < pages:
            time.sleep(random.uniform(1, 2))
    
    # Limit the number of links to process
    return article_links[:max_links]

def crawl_and_extract_cafef(category_url, max_links=100, pages=10, output_file="cafef_articles.json"):
    """Crawl links from a category page and extract article content from each"""
    # Get article links
    logger.info(f"Crawling links from: {category_url} (up to {pages} pages)")
    article_links = crawl_cafef_links(category_url, max_links, pages)
    logger.info(f"Found {len(article_links)} article links")
    
    # Extract articles
    articles = []
    for i, link in enumerate(article_links):
        try:
            logger.info(f"Processing article {i+1}/{len(article_links)}: {link}")
            
            article_data = extract_cafef_article(link)
            if article_data:
                articles.append(article_data)
                # Use simple logging for titles to avoid encoding issues
                logger.info(f"Successfully extracted article {i+1}")
            else:
                logger.warning(f"Failed to extract article from: {link}")
            
            # Save intermediate results periodically (every 10 articles)
            if (i + 1) % 10 == 0:
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(articles, f, ensure_ascii=False, indent=4)
                    logger.info(f"Saved intermediate results ({len(articles)} articles) to {output_file}")
                except Exception as e:
                    logger.error(f"Error saving intermediate results: {e}")
            
            # Polite delay to avoid overloading the server
            time.sleep(random.uniform(1, 3))
            
        except Exception as e:
            logger.error(f"Error processing article {link}: {e}")
            continue
    
    # Save final results to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved {len(articles)} articles to {output_file}")
    except Exception as e:
        logger.error(f"Error saving to JSON: {e}")
        
        # Emergency backup - try to save with a different filename
        try:
            backup_file = output_file.replace('.json', '_backup.json')
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved backup to {backup_file}")
        except Exception as e2:
            logger.error(f"Error saving backup: {e2}")
    
    return articles

if __name__ == "__main__":
    # Fix console encoding for Windows
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # The URL of the CafeF stock market news page
    category_url = "https://cafef.vn/thi-truong-chung-khoan.chn"
    
    # Number of articles to extract and pages to crawl
    max_articles = 10000000 # Adjust this number as needed
    pages_to_crawl = 2500    # Adjust this number as needed
    
    # Output file
    output_file = "cafef_stock_articles.json"
    
    # Run the crawling and extraction
    articles = crawl_and_extract_cafef(category_url, max_articles, pages_to_crawl, output_file)
    