"""Verify all files listed in a released package's SHA256SUMS.txt."""
from pathlib import Path
import argparse,hashlib

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();count=0
    root=a.root.resolve()
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        expected,relative=line.split('  ',1);path=(root/relative).resolve()
        if not path.is_relative_to(root):raise ValueError('Invalid relative path')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=expected:raise ValueError('Checksum failed: '+relative)
        count+=1
    print(f'PASS: {count} files verified in {root}')

if __name__=='__main__':main()
