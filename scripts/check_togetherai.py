import os
import requests
from dotenv import load_dotenv

# Load your exact API key
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, "ATT05522.env")
load_dotenv(env_path)

api_key = os.getenv("TOGETHER_API_KEY")
url = "https://api.together.xyz/v1/models"
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {api_key}"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    models = response.json()
    print("Available Serverless Instruct/Chat Models on Together AI:\n")
    
    for m in models:
        model_id = m.get("id", "")
        pricing = m.get("pricing", {})
        
        # 1. Filter for chat/instruct tuned models
        is_chat_model = any(kw in model_id.lower() for kw in ["instruct", "chat", "-it"])
        
        # 2. Filter for Serverless (Hourly cost is 0, or it explicitly has an input cost, or is Free)
        # We also check for the "-Turbo" and "-Free" naming conventions Together uses
        has_token_price = pricing.get("input", 0) > 0 or pricing.get("base", 0) > 0
        is_free_serverless = "free" in model_id.lower() or "turbo" in model_id.lower()
        has_no_hourly = pricing.get("hourly", 0) == 0
        
        is_serverless = (has_token_price or is_free_serverless) and has_no_hourly
        
        if is_chat_model and is_serverless:
            print(f'"{model_id}"')
            
else:
    print(f"Failed to fetch models: {response.status_code}")