"""Download the corrected BalOCR v2 data without loading it all into memory.

Install huggingface_hub, then:
  python download_ocr_release.py --output BalOCR-data
"""
import argparse,hashlib,json,os
from pathlib import Path
import shutil,zipfile
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
from huggingface_hub import HfApi,hf_hub_download

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def extract_checked(path,destination):
    destination=destination.resolve()
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist())>2_000_000_000:raise ValueError('Expansion limit exceeded')
        for i in z.infolist():
            if not (destination/i.filename).resolve().is_relative_to(destination):raise ValueError('Unsafe member')
            if ((i.external_attr>>16)&0o170000)==0o120000:raise ValueError('Symlink not allowed')
        z.extractall(destination)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--revision');p.add_argument('--complete',action='store_true')
    a=p.parse_args();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    repo='shah-bakhsh/BalDataset';revision=a.revision or HfApi().repo_info(repo,repo_type='dataset').sha
    cache=root/'download_cache'
    get=lambda name,rev:Path(hf_hub_download(repo,name,repo_type='dataset',revision=rev,local_dir=cache))
    name='ocr/canonical/release.json' if a.complete else 'ocr/v2/release.json'
    manifest=json.loads(get(name,revision).read_text(encoding='utf8'))
    base=manifest if a.complete else json.loads(get(manifest['base_manifest_path'],manifest['base_revision']).read_text(encoding='utf8'))
    part_revision=revision if a.complete else manifest['base_revision']
    archive=root/('BalOCR-Dataset-v2.zip' if a.complete else 'base_images.zip')
    with archive.open('wb') as out:
        for part in base['parts']:
            path=get(part['path'],part_revision)
            if digest(path)!=part['sha256']:raise ValueError('Part checksum mismatch')
            with path.open('rb') as f:shutil.copyfileobj(f,out,1024*1024)
    expected=manifest['archive_sha256'] if a.complete else manifest['base_archive_sha256']
    if digest(archive)!=expected:raise ValueError('Base archive checksum mismatch')
    extracted=root if a.complete else root/'prepared'
    extract_checked(archive,extracted)
    if not a.complete:
        overlay=get(manifest['overlay_path'],revision)
        if digest(overlay)!=manifest['overlay_sha256']:raise ValueError('Overlay checksum mismatch')
        extract_checked(overlay,extracted)
    target=root/'data/prepared' if a.complete else extracted
    report=json.loads((target/'training_input_report.json').read_text(encoding='utf8'))
    if report['training_records']!=manifest['training_records']:raise ValueError('Count mismatch')
    for item in manifest.get('provenance_files',[]):
        path=(extracted/item['path']).resolve()
        if not path.is_relative_to(extracted.resolve()) or not path.is_file() or digest(path)!=item['sha256']:
            raise ValueError('Missing or changed source attribution/license file')
    for filename in ['train.jsonl','evaluation.jsonl']:
        count=0
        with (target/filename).open(encoding='utf8') as f:
            for line in f:
                row=json.loads(line)
                count+=1
                if row.get('attribution_reference') and not (target/row['attribution_reference']).is_file():
                    raise ValueError('Broken record attribution reference')
        expected_count=report['training_records' if filename=='train.jsonl' else 'evaluation_records']
        if count!=expected_count:raise ValueError('Actual manifest row count mismatch')
    print(json.dumps({'revision':revision,'prepared':str(target),'training_records':report['training_records'],
                      'evaluation_records':report['evaluation_records'],'checksums':'PASS'},indent=2))

if __name__=='__main__':main()
