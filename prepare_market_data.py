"""Download, verify and extract the market data release (standard library only)."""
from pathlib import Path
import argparse,hashlib,json,os,tarfile,urllib.request

ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()

def prepare(archive,dest,download=True):
    manifest=json.loads((ROOT/'data/market_data_manifest.json').read_text())
    if not archive.is_file():
        if not download:raise FileNotFoundError(archive)
        archive.parent.mkdir(parents=True,exist_ok=True)
        temp=archive.with_name(archive.name+'.download.tmp')
        try:
            with urllib.request.urlopen(manifest['download_url'],timeout=60) as source,temp.open('wb') as out:
                while block:=source.read(1024*1024):out.write(block)
            if digest(temp)!=manifest['archive_sha256']:raise ValueError('Downloaded archive checksum mismatch')
            os.replace(temp,archive)
        finally:temp.unlink(missing_ok=True)
    if archive.stat().st_size!=manifest['archive_bytes'] or digest(archive)!=manifest['archive_sha256']:
        raise ValueError('Archive checksum mismatch')
    expected={x['path']:x for x in manifest['files']}
    seen=set()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            path=Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts or member.name not in expected or member.name in seen:
                raise ValueError('Unexpected archive entry: '+member.name)
            row=expected[member.name];seen.add(member.name)
            target=dest/path
            if target.exists():
                if target.stat().st_size==row['bytes'] and digest(target)==row['sha256']:continue
                raise FileExistsError('Destination differs from release: '+str(target))
            target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_name(target.name+'.restore.tmp');h=hashlib.sha256();size=0
            try:
                with tar.extractfile(member) as source,temp.open('wb') as out:
                    while block:=source.read(1024*1024):h.update(block);size+=len(block);out.write(block)
                if size!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('Data checksum mismatch: '+member.name)
                os.replace(temp,target)
            finally:temp.unlink(missing_ok=True)
    if seen!=set(expected):raise ValueError('Incomplete market data archive')
    print(f'Verified and prepared {len(seen)} data files at {dest}')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=ROOT/'data/treasury-etf-market-data-v1.tar.gz')
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/market_data')
    p.add_argument('--no-download',action='store_true')
    a=p.parse_args();prepare(a.archive,a.output_dir,not a.no_download)
