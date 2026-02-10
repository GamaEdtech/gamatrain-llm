import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import re
import time

# Configuration
SITEMAP_URLS = [
    "https://gamatrain.com/sitemap/blog-1.xml",
    "https://gamatrain.com/sitemap/blog-2.xml"
]
OUTPUT_FILE = "gamatrain_finetune_data.jsonl"
DELAY_BETWEEN_REQUESTS = 3
MAX_BLOGS = 5000

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return text

def fetch_url(url):
    """Fetch content from a URL."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 200:
                return response.read().decode('utf-8')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None

def extract_blog_urls_from_sitemap(sitemap_url):
    """Extract all blog URLs from a sitemap XML."""
    print(f"Fetching sitemap: {sitemap_url}")
    content = fetch_url(sitemap_url)
    if not content:
        return []
    
    urls = []
    try:
        root = ET.fromstring(content)
        ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        for url_elem in root.findall('.//sm:url', ns):
            loc = url_elem.find('sm:loc', ns)
            if loc is not None and loc.text:
                urls.append(loc.text)
    except Exception as e:
        print(f"Error parsing sitemap: {e}")
    
    return urls

def extract_simple_blog_content(blog_url):
    """Extract title and simplified content from blog page using basic regex."""
    print(f"Fetching: {blog_url}")
    html = fetch_url(blog_url)
    if not html:
        return None, None
    
    # Extract title from <title> tag
    title_match = re.search(r'<title[^>]*>([^<]+)', html, re.IGNORECASE)
    if title_match:
        title = clean_text(title_match.group(1))
        title = re.sub(r'\s*[-|]\s*Gamatrain.*$', '', title, flags=re.IGNORECASE).strip()
    else:
        return None, None
    
    # Simple approach: Extract all text between <p> tags
    paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL | re.IGNORECASE)
    
    # Clean paragraphs and combine
    clean_paragraphs = []
    for p in paragraphs:
        # Remove HTML tags
        clean_p = re.sub(r'<[^>]+>', '', p)
        clean_p = clean_text(clean_p)
        if len(clean_p) > 30:  # Only keep substantial paragraphs
            clean_paragraphs.append(clean_p)
    
    if not clean_paragraphs:
        return None, None
    
    # Combine first 3-4 paragraphs (limiting to ~500 words)
    content = ' '.join(clean_paragraphs[:4])
    words = content.split()[:500]
    content = ' '.join(words)
    
    return title, content

def format_for_qwen(title, content):
    """Format blog data for Qwen fine-tuning."""
    if not title or not content or len(content) < 100:
        return None
    
    messages = [
        {"role": "system", "content": "You are Gamatrain AI, an intelligent educational assistant. Answer questions based on the provided educational content."},
        {"role": "user", "content": f"Tell me about {title}"},
        {"role": "assistant", "content": content}
    ]
    return {"messages": messages}

def main():
    print("Starting Blog Data Extraction...")
    blog_entries = []
    
    # Extract all blog URLs from sitemaps
    all_blog_urls = []
    for sitemap_url in SITEMAP_URLS:
        urls = extract_blog_urls_from_sitemap(sitemap_url)
        all_blog_urls.extend(urls)
        print(f"Found {len(urls)} blog URLs in {sitemap_url}")
        time.sleep(1)
    
    print(f"\nTotal blog URLs found: {len(all_blog_urls)}")
    print(f"Extracting first {MAX_BLOGS} blogs...\n")
    
    # Fetch content from each blog (limited to MAX_BLOGS)
    for i, blog_url in enumerate(all_blog_urls[:MAX_BLOGS], 1):
        print(f"[{i}/{MAX_BLOGS}] ", end='')
        title, content = extract_simple_blog_content(blog_url)
        
        if title and content:
            entry = format_for_qwen(title, content)
            if entry:
                blog_entries.append(entry)
                print(f"✓ Extracted: {title[:50]}")
            else:
                print(f"✗ Skipped (too short)")
        else:
            print(f"✗ Failed to extract content")
        
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    print(f"\n\nTotal blog entries successfully extracted: {len(blog_entries)}")
    
    if blog_entries:
        # Append to existing dataset
        print(f"Appending {len(blog_entries)} blog entries to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for entry in blog_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print("✅ Done! Blog data appended to dataset.")
    else:
        print("❌ No blog data extracted.")

if __name__ == "__main__":
    main()
