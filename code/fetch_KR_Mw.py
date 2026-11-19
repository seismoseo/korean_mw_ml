"""Download the Hong and Sheen (2026) KR_Mw catalog at the pinned commit and check its checksums.

The two CSV files are not redistributed here because the KR_Mw repository has no license.
compare_KR_Mw.py, hong_regressions.py, report_Hong_regressions.py and the potency scripts read them
from inputs/KR_Mw/. The commit, URLs and SHA-256 values are in inputs/KR_Mw/source.json.

    python code/fetch_KR_Mw.py           # download missing files, then verify both
"""
from pathlib import Path
import hashlib, json, urllib.request

R = Path(__file__).resolve().parents[1]
folder = R / 'inputs/KR_Mw'
source = json.loads((folder / 'source.json').read_text())
for item in source['files']:
    path = folder / item['file']
    if not path.exists():
        url = item['url'].replace('github.com/', 'raw.githubusercontent.com/').replace('/blob/', '/')
        print('downloading', url, flush=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item['sha256']:
        raise SystemExit(f'{path.name}: SHA-256 {digest} does not match the pinned {item["sha256"]}')
    print('verified', path.name, item['bytes'], 'bytes')
