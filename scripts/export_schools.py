"""
Export schools from Gamatrain API to custom_docs.json
Run this script to fetch all schools and add them to RAG
"""

import requests
import json
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration - Update these values
API_BASE_URL = "https://api.gamaedtech.com/api/v1"
AUTH_TOKEN = ""  # Add your token if needed

OUTPUT_FILE = "../data/custom_docs.json"


def fetch_all_schools():
    """Fetch all schools from API with pagination."""
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}
    all_schools = []
    skip = 0
    batch_size = 100
    
    print("Fetching schools from API...")
    
    while True:
        params = {
            "PagingDto.PageFilter.Size": batch_size,
            "PagingDto.PageFilter.Skip": skip,
            "PagingDto.PageFilter.ReturnTotalRecordsCount": "true"
        }
        
        try:
            resp = requests.get(
                f"{API_BASE_URL}/schools",
                params=params,
                headers=headers,
                verify=False,
                timeout=30
            )
            
            if resp.status_code != 200:
                print(f"Error: {resp.status_code}")
                break
            
            data = resp.json()
            schools = data.get("data", {}).get("list", [])
            
            if not schools:
                break
            
            all_schools.extend(schools)
            print(f"  Fetched {len(all_schools)} schools...")
            
            if len(schools) < batch_size:
                break
            
            skip += batch_size
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    return all_schools


def convert_to_documents(schools):
    """Convert schools to document format for RAG."""
    documents = []
    
    for school in schools:
        name = school.get("name", "")
        
        # Skip test/gamatrain schools
        if not name or "gamatrain" in name.lower():
            continue
        
        # Build comprehensive text
        text_parts = [f"School Name: {name}"]
        
        if school.get("cityTitle"):
            text_parts.append(f"City: {school['cityTitle']}")
        
        if school.get("countryTitle"):
            text_parts.append(f"Country: {school['countryTitle']}")
        
        if school.get("description"):
            text_parts.append(f"Description: {school['description']}")
        
        if school.get("address"):
            text_parts.append(f"Address: {school['address']}")
        
        if school.get("phone"):
            text_parts.append(f"Phone: {school['phone']}")
        
        if school.get("website"):
            text_parts.append(f"Website: {school['website']}")
        
        if school.get("studentCount"):
            text_parts.append(f"Students: {school['studentCount']}")
        
        documents.append({
            "text": "\n".join(text_parts),
            "type": "school",
            "id": f"school_{school.get('id', len(documents))}"
        })
    
    return documents


def update_custom_docs(new_documents):
    """Add new documents to custom_docs.json."""
    
    # Load existing
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"documents": []}
    
    # Remove old school documents
    data["documents"] = [d for d in data["documents"] if d.get("type") != "school"]
    
    # Add new school documents
    data["documents"].extend(new_documents)
    
    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(new_documents)} schools to {OUTPUT_FILE}")


def main():
    # Fetch schools
    schools = fetch_all_schools()
    print(f"\nTotal schools fetched: {len(schools)}")
    
    if not schools:
        print("No schools found. Check API URL and credentials.")
        return
    
    # Convert to documents
    documents = convert_to_documents(schools)
    print(f"Converted to {len(documents)} documents")
    
    # Update custom_docs.json
    update_custom_docs(documents)
    
    print("\nDone! Now refresh the RAG index:")
    print("  curl -X POST http://localhost:8002/v1/refresh")


if __name__ == "__main__":
    main()
