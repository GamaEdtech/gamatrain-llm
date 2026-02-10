import urllib.request
import urllib.error
import urllib.parse
import json
import os
import re
import time
from typing import List, Dict, Any

# Configuration
API_BASE_URL = "https://core.gamatrain.com"
OUTPUT_FILE = "gamatrain_finetune_data.jsonl"
MAX_PAGES = 20  # Increased limit to get more data
DELAY_BETWEEN_REQUESTS = 1

# Endpoints
ENDPOINTS = {
    "papers": {"url": "https://core.gamatrain.com/api/v1/search", "params": {"type": "test"}},
    "multimedia": {"url": "https://core.gamatrain.com/api/v1/search", "params": {"type": "learnfiles"}},
    "quizhub": {"url": "https://core.gamatrain.com/api/v1/search", "params": {"type": "azmoon"}},
    # "forum": {"url": "https://core.gamatrain.com/api/v1/search", "params": {"type": "question"}}, # Disabled due to low quality
    "tutorials": {"url": "https://core.gamatrain.com/api/v1/search", "params": {"type": "dars"}},
    "schools": {
        "url": "https://api.gamaedtech.com/api/v1/schools",
        "params": {
            "PagingDto.PageFilter.Size": 20,
            "PagingDto.PageFilter.ReturnTotalRecordsCount": "true",
            "PagingDto.SortFilter[0].sortType": "Desc",
            "PagingDto.SortFilter[0].column": "lastModifyDate"
        },
        "pagination_type": "skip_size"
    }
}

# Headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

# Try to load token
TOKEN_FILE = os.path.join(os.path.dirname(__file__), "token.txt")
if os.path.exists(TOKEN_FILE):
    with open(TOKEN_FILE, 'r') as f:
        token = f.read().strip()
        if token:
            HEADERS["Authorization"] = f"Bearer {token}"
            print("Loaded API token from token.txt")

def clean_html(html_content: str) -> str:
    """Remove HTML tags and clean whitespace."""
    if not html_content:
        return ""
    text = re.sub(r'<[^>]+>', ' ', str(html_content))
    text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_for_qwen(instruction: str, input_text: str, output_text: str) -> Dict[str, Any]:
    """Format data into Qwen/OpenAI chat format."""
    messages = [
        {"role": "system", "content": "You are Gamatrain AI, an intelligent educational assistant. Answer questions based on the provided educational content."},
        {"role": "user", "content": f"{instruction}\n\nContext:\n{input_text}" if input_text else instruction},
        {"role": "assistant", "content": output_text}
    ]
    return {"messages": messages}

def make_request(full_url: str, params: Dict = None) -> Dict:
    """Make HTTP GET request."""
    if params:
        query_string = urllib.parse.urlencode(params)
        full_url += f"?{query_string}"
    
    try:
        req = urllib.request.Request(full_url, headers=HEADERS)
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        print(f"Error fetching {full_url}: {e}")
        try:
            print(e.read().decode('utf-8'))
        except:
            pass
    except Exception as e:
        print(f"Unexpected error fetching {full_url}: {e}")
    return {}

def fetch_search_category(category_name: str, config: Dict) -> List[Dict]:
    """Fetch data from search-based endpoints with pagination."""
    print(f"Fetching {category_name}...")
    dataset = []
    page = 1
    
    while page <= MAX_PAGES:
        params = config["params"].copy()
        
        if config.get("pagination_type") == "skip_size":
            params["PagingDto.PageFilter.Skip"] = (page - 1) * params["PagingDto.PageFilter.Size"]
        else:
            params["page"] = page
        
        data = make_request(config["url"], params)
        
        # Handle response structure
        items = []
        if isinstance(data, dict):
            if 'data' in data and isinstance(data['data'], dict) and 'list' in data['data']:
                items = data['data']['list']
            elif 'data' in data and isinstance(data['data'], list):
                items = data['data']
            elif 'list' in data:
                items = data['list']
            elif 'items' in data:
                 items = data['items']
        elif isinstance(data, list):
            items = data
        
        if not items:
            break
            
        print(f"  Page {page}: Found {len(items)} items")
        
        for item in items:
            try:
                entry = process_item(category_name, item)
                if entry:
                    dataset.append(entry)
            except Exception as e:
                pass # Skip malformed items

        page += 1
        time.sleep(DELAY_BETWEEN_REQUESTS)
        
    return dataset

def process_item(category: str, item: Dict) -> Dict:
    """Process a single item based on its category."""
    title = clean_html(item.get('title', '') or item.get('schoolTitle', '') or item.get('name', ''))
    description = clean_html(item.get('description', '') or item.get('summary', '') or item.get('content', ''))
    
    if not title:
        return None

    instruction = ""
    output = ""

    if category == "papers":
        instruction = f"Does Gamatrain have past papers for {title}?"
        output = f"Yes, Gamatrain offers past papers for {title}. {description}"
    
    elif category == "multimedia":
        instruction = f"What multimedia resources are available for {title}?"
        output = f"We have multimedia content for {title}. {description}"
        
    elif category == "quizhub":
        instruction = f"Tell me about the quiz '{title}'."
        output = f"The quiz '{title}' is available on QuizHub. {description}"
        
    elif category == "forum":
        instruction = title
        output = description if description else "Please check the forum link for the detailed answer."
        
    elif category == "tutorials":
        instruction = f"Explain the topic '{title}'."
        output = description if description else f"This is a tutorial about {title}."
        
    elif category == "schools":
        region = item.get('regionTitle', '') or item.get('region', {}).get('title', '')
        state = item.get('stateTitle', '') or item.get('state', {}).get('title', '')
        city = item.get('cityTitle', '') or item.get('city', {}).get('title', '')
        
        location_parts = [p for p in [city, region, state] if p]
        location_str = ", ".join(location_parts)
        
        instruction = f"Tell me about {title} school."
        output = f"{title} is a school located in {location_str}." if location_str else f"{title} is a school listed on Gamatrain."

    if instruction and output:
        return format_for_qwen(instruction, "", output)
    return None

def main():
    print("Starting Comprehensive Gamatrain Data Extraction...")
    all_data = []
    
    for category, config in ENDPOINTS.items():
        try:
            category_data = fetch_search_category(category, config)
            all_data.extend(category_data)
            print(f"Total {category} extracted: {len(category_data)}")
        except Exception as e:
            print(f"Failed to extract {category}: {e}")

    print(f"\nTotal items extracted: {len(all_data)}")
    
    if all_data:
        print(f"Writing to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print("Done!")
    else:
        print("No data extracted. Check your token or network connection.")

if __name__ == "__main__":
    main()
