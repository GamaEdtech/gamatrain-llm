import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_BASE_URL = "https://185.204.170.142/api/v1"
AUTH_TOKEN = "1735%7CCfDJ8AHj4VWfmz9DpMpNxts7109iJyV5YLZVw3PwbvKW5DKqAEgJJH9q%2FbrwZH5%2Bea87uMdj4LXj58uTZ7snP8YcRP36uezVDspGvzUhEQTQ5Du4icTip2mah0Cq4C86s%2Bpy31PAxl%2FpsRIJXlugy7EmHgSgq9sOgSW9YPr%2BB1Pf2gdT4umedbopK1a0%2F6YKPrBL2Q9%2BNM2XzeBSmFcgXEvsT5rP28t%2BUIC2veZU99lS2849"

headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}

# لیست endpointها برای تست
endpoints = [
    ("GET", "/blogs/posts", {"PagingDto.PageFilter.Size": 10}),
    ("GET", "/questions", {"Count": 10}),
    ("GET", "/identities/profiles", None),
    ("GET", "/identities/authenticated", None),
    ("GET", "/grades", None),
    ("GET", "/boards", None),
    ("GET", "/subjects", None),
    ("GET", "/schools", {"PagingDto.PageFilter.Size": 10}),
    ("GET", "/tags/Post", None),
]

print("تست دسترسی به endpointها:\n")
print("-" * 60)

for method, endpoint, params in endpoints:
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            resp = requests.get(url, params=params, headers=headers, verify=False, timeout=10)
        
        status = "✅" if resp.status_code == 200 else "❌"
        print(f"{status} {endpoint}: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get("succeeded"):
                # نمایش تعداد رکوردها اگه لیست باشه
                result_data = data.get("data")
                if isinstance(result_data, dict) and "list" in result_data:
                    print(f"   → {len(result_data['list'])} رکورد")
                elif isinstance(result_data, list):
                    print(f"   → {len(result_data)} رکورد")
    except Exception as e:
        print(f"❌ {endpoint}: {e}")

print("-" * 60)
