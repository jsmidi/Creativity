import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load the file
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, "ATT05522.env")
load_dotenv(env_path)

# 2. Get your Together key (assuming it is named TOGETHER_API_KEY in your .env)
together_key = os.getenv("TOGETHER_API_KEY")

if not together_key:
    print("Error: TOGETHER_API_KEY not found in the file.")
    exit()

# 3. Point the client DIRECTLY to Together AI, not OpenRouter
client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=together_key
)

try:
    print("Testing connection directly to Together AI...")
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo", 
        messages=[{"role": "user", "content": "Write a one-sentence poem about a brick."}],
        max_tokens=50
    )
    print("\nSuccess! The API is working perfectly.")
    print("-" * 40)
    print(response.choices[0].message.content.strip())
    print("-" * 40)
    
except Exception as e:
    print(f"\nAPI Error occurred: {e}")