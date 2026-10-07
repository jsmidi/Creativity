"""List provider model IDs without loading local weights or generating responses.

Example: python scripts/check_providers.py together --contains Qwen
Credentials are loaded only when main runs, from scripts/ATT05522.env.
"""
import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

URLS = {
    'together': 'https://api.together.xyz/v1/models',
    'groq': 'https://api.groq.com/openai/v1/models',
    'openrouter': 'https://openrouter.ai/api/v1/models',
}


def fetch_models(provider):
    """Return the provider's model list; authenticated catalogs require an API key."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name('ATT05522.env'))
    key = os.getenv(f'{provider.upper()}_API_KEY')
    if provider != 'openrouter' and not key:
        raise ValueError(f'Missing {provider.upper()}_API_KEY')
    headers = {'Accept': 'application/json'}
    if key:
        headers['Authorization'] = f'Bearer {key}'
    request = Request(URLS[provider], headers=headers)
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return payload if isinstance(payload, list) else payload.get('data', [])


def matches(model, contains='', free_only=False):
    """Filter IDs by a case-insensitive substring and optional zero token prices."""
    if contains.lower() not in model.get('id', '').lower():
        return False
    if not free_only:
        return True
    pricing = model.get('pricing', {})
    return str(pricing.get('prompt')) == '0' and str(pricing.get('completion')) == '0'


def main():
    """Print matching model IDs for one explicit provider; make no generation calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('provider', choices=URLS)
    parser.add_argument('--contains', default='')
    parser.add_argument('--free-only', action='store_true', help='Require zero prompt and completion prices')
    args = parser.parse_args()
    for model in fetch_models(args.provider):
        if matches(model, args.contains, args.free_only):
            print(model['id'])


if __name__ == '__main__':
    main()
