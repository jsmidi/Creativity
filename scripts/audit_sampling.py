"""Export recorded sampling settings; absent parameters remain unknown."""
import argparse
import csv
import hashlib
from pathlib import Path


def audit(root):
    """Yield one row per distinct recorded sampling tuple in each raw CSV."""
    for path in sorted(root.rglob('*.csv')):
        with path.open() as handle:
            reader = csv.DictReader(handle)
            if not {'Response', 'Temperature'}.issubset(reader.fieldnames or []):
                continue
            counts = {}
            for row in reader:
                key = tuple(row.get(k, 'unknown_not_recorded') for k in
                            ['Temperature', 'Top_P', 'Top_K'])
                counts[key] = counts.get(key, 0) + 1
        for key, count in counts.items():
            yield dict(Source=str(path), SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       Temperature=key[0], Top_P=key[1], Top_K=key[2], Responses=count)


def main():
    """Write an evidence-only CSV audit of an outputs directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs'))
    parser.add_argument('--output', type=Path, default=Path('analysis/sampling_audit.csv'))
    args = parser.parse_args()
    rows = list(audit(args.root))
    if not rows:
        parser.error('No raw response files with temperature records found')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} sampling records to {args.output}')


if __name__ == '__main__':
    main()
