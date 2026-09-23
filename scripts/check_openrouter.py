import requests

url = "https://openrouter.ai/api/v1/models"
response = requests.get(url)

if response.status_code == 200:
    models = response.json().get("data", [])
    print("Currently Active Free Models on OpenRouter:\n")
    
    # We filter the giant list down to ONLY models where the price is exactly $0.00
    for m in models:
        pricing = m.get("pricing", {})
        if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
            print(f'"{m["id"]}"')
else:
    print(f"Failed to fetch: {response.status_code}")