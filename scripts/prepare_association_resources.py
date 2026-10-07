"""Download WordNet once and freeze a noun vocabulary for exploratory CDAT/DRAT."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

URL = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip'


def main():
    """Save sorted single-word noun lemmas, upstream license and download hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('databases/association'))
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory exists; preserve the frozen resources or choose another path')
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read()
    archive = zipfile.ZipFile(io.BytesIO(data))
    words = sorted({line.split()[0] for line in archive.read('wordnet/index.noun').decode().splitlines()
                    if line and not line[0].isspace() and re.fullmatch('[a-z]+(?:-[a-z]+)*', line.split()[0])})
    args.output_dir.mkdir(parents=True)
    path = args.output_dir/'nouns.txt'
    path.write_text('\n'.join(words)+'\n')
    (args.output_dir/'WORDNET_LICENSE').write_bytes(archive.read('wordnet/LICENSE'))
    (args.output_dir/'source.json').write_text(json.dumps(dict(url=URL,
        archive_sha256=hashlib.sha256(data).hexdigest(), nouns_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        count=len(words), filtering='unique lowercase single-word noun lemmas, hyphens allowed'), indent=2))
    print(f'Saved {len(words)} nouns to {path}')


if __name__ == '__main__':
    main()
