import os
import requests
from dotenv import load_dotenv

# Load your Groq API key from your env file
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, "ATT05522.env") # Adjust if your file is just named .env
load_dotenv(env_path)

api_key = os.getenv("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/models"
headers = {
    "Authorization": f"Bearer {api_key}"
}

print("Fetching available models from Groq...\n")
response = requests.get(url, headers=headers)

if response.status_code == 200:
    # Groq correctly uses the 'data' wrapper, so parsing is easy
    models = response.json().get("data", [])
    for m in models:
        print(f'"{m["id"]}"')
else:
    print(f"Failed to fetch models: {response.status_code} - {response.text}")