"""Restore the frozen training layout from the two downloaded Kaggle packages."""
from pathlib import Path
import argparse,hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    a=p.parse_args();roots={'data':a.data_root,'weights':a.weights_root}
    for row in json.loads((ROOT/'reproduction/DEFERRED_FILES.json').read_text()):
        source=roots[row['source_package']]/row['source'];dest=ROOT/row['destination']
        if sha(source)!=row['sha256']:raise ValueError('Source checksum failed: '+str(source))
        if dest.exists():
            if sha(dest)!=row['sha256']:raise ValueError('Existing file differs: '+str(dest))
        else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    for tag in ['legacy','nafnet']:
        root=ROOT/'reproduction'/tag
        inventory=json.loads((root/'audit/SCIENCE_FILE_SHA256.json').read_text())['files']
        for relative,expected in inventory.items():
            if sha(root/relative)!=expected:raise ValueError('Frozen source differs: '+relative)
        print(tag+': frozen training source and assets verified')

if __name__=='__main__':main()
