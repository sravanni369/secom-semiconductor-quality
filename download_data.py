"""Download the original UCI files; retain attribution and SHA-256 provenance."""
from pathlib import Path
import hashlib
import io
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
URL = 'https://archive.ics.uci.edu/static/public/179/secom.zip'

def main():
    folder = ROOT / 'data'
    folder.mkdir(exist_ok=True)
    payload = urllib.request.urlopen(URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in ('secom.data', 'secom_labels.data', 'secom.names'):
            (folder / name).write_bytes(archive.read(name))
    manifest = {'url': URL, 'citation': 'McCann, M., & Johnston, A. (2008). SECOM. https://doi.org/10.24432/C54305',
                'license': 'CC BY 4.0', 'archive_sha256': hashlib.sha256(payload).hexdigest(),
                'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('secom.*')}}
    manifest['files']['secom_labels.data'] = hashlib.sha256((folder/'secom_labels.data').read_bytes()).hexdigest()
    (folder/'provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
