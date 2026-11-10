"""Audit and repair the supplied archive without changing its labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import unicodedata
import zipfile

from PIL import Image

PREFIX = 'balochi-ocr-dataset/'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def validate_member(name):
    name = name.replace('\\', '/')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or ':' in name or '\x00' in name or not p.parts:
        raise ValueError(f'Unsafe archive member: {name!r}')
    return str(p)

def resolve_member(name, occurrence):
    p = PurePosixPath(validate_member(name))
    return str(p.with_name(f'{p.stem}__dup{occurrence}{p.suffix}')) if occurrence else str(p)

def text_key(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())

def partition_pairs(rows):
    by_image = defaultdict(set)
    for r in rows:
        by_image[r['image_sha256']].add(text_key(r['text']))
    good, bad, seen = [], [], set()
    for r in rows:
        key = (r['image_sha256'], text_key(r['text']))
        if len(by_image[r['image_sha256']]) > 1:
            bad.append({**r, 'quarantine_reason': 'same_image_conflicting_transcriptions'})
        elif key in seen:
            bad.append({**r, 'quarantine_reason': 'duplicate_image_and_text'})
        else:
            seen.add(key)
            good.append(r)
    return good, bad

def assign_splits(rows):
    """Date proxies and exact/near duplicate groups; unknown dates train only.

    These are diagnostic splits. Dates do not prove independent source books.
    """
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = root(i), root(j)
        parent[max(a,b)] = min(a,b)
    keys = {}
    for i, r in enumerate(rows):
        these = [('image', r['image_sha256']), ('text', text_key(r['text']))]
        if r.get('capture_date'):
            these.append(('date', r['capture_date']))
        for key in these:
            if key in keys:
                union(i, keys[key])
            else:
                keys[key] = i
    # Approximate candidate search, followed by exact shingle Jaccard check.
    # Short repeated phrases are intentionally not treated as whole documents.
    index = defaultdict(list)
    def word_shingles(text):
        words = text_key(text).split()
        return {hashlib.blake2b(' '.join(words[n:n+5]).encode(), digest_size=8).digest()
                for n in range(len(words)-4)}
    for i, r in enumerate(rows):
        words = text_key(r['text']).split()
        if len(words) < 40:
            continue
        s = word_shingles(r['text'])
        sketch = sorted(s)[:8]
        candidates = set(j for key in sketch for j in index[key])
        for hit in candidates:
            other = word_shingles(rows[hit]['text'])
            if len(s & other) / len(s | other) >= 0.85:
                union(i, hit)
        for key in sketch:
            index[key].append(i)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[root(i)].append(i)
    for indices in groups.values():
        ids = sorted(rows[i]['id'] for i in indices)
        group = 'arabic_' + sha('\n'.join(ids).encode())[:20]
        unknown = any(not rows[i].get('capture_date') for i in indices)
        split = 'train' if unknown or int(sha(group.encode())[:8], 16) % 100 >= 15 else 'validation'
        for i in indices:
            rows[i].update(split=split, split_group=group,
                           split_basis='date_proxy_plus_image_text_duplicate_groups',
                           independent_book_holdout_verified=False)
    return rows

def prepare_arabic(archive, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output/'images'/'arabic'
    image_dir.mkdir(parents=True, exist_ok=True)
    quarantine, valid, occurrence = [], [], Counter()
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if sum(i.file_size for i in infos) > 2_000_000_000:
            raise ValueError('Archive exceeds 2 GB expansion limit')
        for info in infos:
            validate_member(info.filename)
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Archive contains symlink')
            if info.file_size > 50_000_000:
                raise ValueError('Archive member exceeds 50 MB limit')
        names = set(z.namelist())
        if len(names)!=len(infos):
            raise ValueError('Duplicate ZIP member names are ambiguous')
        info=json.loads(z.read(PREFIX+'dataset_info.json').decode('utf-8-sig'))
        if info.get('license','').lower()!='cc-by-4.0':
            raise ValueError('Archive license declaration is not the approved CC-BY-4.0 source')
        source = output/'source_metadata'
        source.mkdir(exist_ok=True)
        for name in ['dataset_info.json', 'README.txt', 'metadata.jsonl', 'metadata.csv']:
            (source/name).write_bytes(z.read(PREFIX+name))
        rows = [json.loads(line) for line in z.read(PREFIX+'metadata.jsonl').decode('utf-8-sig').splitlines() if line.strip()]
        if len({r['id'] for r in rows})!=len(rows):
            raise ValueError('Duplicate source record IDs')
        repairs = 0
        used = set()
        for row in rows:
            raw_name = row['file_name']
            index = occurrence[raw_name]
            occurrence[raw_name] += 1
            member = PREFIX+resolve_member(raw_name, index)
            reason = None
            if not re.fullmatch(r'[A-Za-z0-9_-]+', row['id']):
                reason = 'invalid_record_id'
            elif member not in names:
                reason = 'missing_image'
            elif not isinstance(row['text'], str) or not row['text'].strip():
                reason = 'empty_transcription'
            if reason:
                quarantine.append({**row, 'quarantine_reason': reason})
                continue
            raw = z.read(member)
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    if im.size != (row['width'], row['height']):
                        raise ValueError('declared_dimensions_do_not_match_occurrence_candidate')
                    pixels = im.convert('L')
                    image_hash = sha(str(pixels.size).encode()+pixels.tobytes())
                    pixels.load()
            except Exception as exc:
                quarantine.append({**row, 'quarantine_reason': str(exc)})
                continue
            dest = image_dir/(row['id']+'.png')
            dest.write_bytes(raw)
            used.add(member)
            repairs += int(index > 0)
            date = re.search(r'(2026)[_-]?(\d{2})[_-]?(\d{2})', raw_name)
            valid.append({**row, 'image': dest.relative_to(output).as_posix(),
                          'original_file_name': raw_name,
                          'resolved_archive_member': member,
                          'mapping_method': 'metadata_occurrence_to_dup_suffix_and_dimension_check' if index else 'original_path_and_dimension_check',
                          'mapping_native_reviewed': False,
                          'image_sha256': image_hash, 'png_sha256': sha(raw),
                          'text_sha256': sha(row['text'].encode()),
                          'script': 'arabic', 'synthetic': False,
                          'source_id': 'academy_ocr_v1', 'license': 'CC-BY-4.0',
                          'attribution': 'Balochi Academy NLP Ecosystem / Thaheer Production, Balochi OCR Ground Truth Dataset v1.0 (2026)',
                          'provider_human_verified_claim': True,
                          'project_native_reviewed': False,
                          'capture_date': '-'.join(date.groups()) if date else None})
        clean, excluded = partition_pairs(valid)
        for row in excluded:
            src=(output/row['image']).resolve()
            dest=(output/'quarantine_images'/src.name).resolve()
            if not src.is_relative_to(output.resolve()) or not dest.is_relative_to(output.resolve()):
                raise ValueError('Quarantine path escapes generated output')
            dest.parent.mkdir(exist_ok=True)
            src.replace(dest)
            row['image']=dest.relative_to(output.resolve()).as_posix()
        quarantine.extend(excluded)
        clean = assign_splits(clean)
        write_jsonl(output/'arabic.jsonl', clean)
        write_jsonl(output/'quarantine.jsonl', quarantine)
        png_members = {n for n in names if n.lower().endswith('.png')}
    archive_hash = hashlib.sha256()
    with open(archive, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            archive_hash.update(block)
    report = dict(source_archive=Path(archive).name, archive_sha256=archive_hash.hexdigest(),
                  original_rows=len(rows), available_pngs=len(png_members), repaired_occurrence_links=repairs,
                  dimension_checked_pairs=len(valid), retained_pairs=len(clean),
                  quarantine_reasons=dict(Counter(r['quarantine_reason'] for r in quarantine)),
                  unused_png_members=sorted(png_members-used),
                  split_counts=dict(Counter(r['split'] for r in clean)),
                  retained_whitespace_words=sum(len(r['text'].split()) for r in clean),
                  retained_characters=sum(len(r['text']) for r in clean),
                  source_license_declaration='CC-BY-4.0 in source README and dataset_info.json',
                  project_native_reviewed_pairs=0,
                  evaluation_limitation='Date proxies and duplicate groups are not verified unseen-book holdouts; no real Latin scan benchmark yet.')
    (output/'arabic_report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='unused_png_members'}, indent=2))
    return clean

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare_arabic(args.archive, args.output)
