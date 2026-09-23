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

print("Fetching available Qwen Instruct models...\n")
response = requests.get(url, headers=headers)

# Parse the raw JSON list directly
if response.status_code == 200:
    models = response.json()
    for m in models:
        # Check if it's a Qwen chat/instruct model
        if "Qwen" in m.get("id", "") and ("Instruct" in m.get("id", "") or "Chat" in m.get("id", "")):
            print(f'"{m["id"]}"')
else:
    print(f"Failed to fetch models: {response.status_code} - {response.text}")