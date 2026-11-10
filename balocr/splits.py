"""Exclude paragraph/crop containment, not just whole-record duplicates."""
from .data import text_key
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .data import write_jsonl

def word_shingles(text):
    words=text_key(text).split()
    return {' '.join(words[n:n+5]) for n in range(len(words)-4)}

def overlaps_reference(training_text,reference_text,threshold=0.85):
    a,b=text_key(training_text),text_key(reference_text)
    if a==b:return True
    # Short punctuation and ordinary shared words are useful OCR training data.
    if min(len(a.split()),len(b.split()))<10:return False
    if a in b or b in a:return True
    sa,sb=word_shingles(a),word_shingles(b)
    return bool(sa and sb and len(sa & sb)/min(len(sa),len(sb))>=threshold)

class ReferenceIndex:
    def __init__(self,records):
        self.references=[]
        for row in records:
            text=text_key(row['text'])
            self.references.append((row['id'],text,len(text.split()),word_shingles(text)))
    def find(self,text,threshold=0.85):
        text=text_key(text);count=len(text.split());shingles=None
        for rid,reference,words,ref_shingles in self.references:
            if text==reference:return rid
            if min(count,words)<10:continue
            if text in reference or reference in text:return rid
            if shingles is None:shingles=word_shingles(text)
            if shingles and ref_shingles and len(shingles & ref_shingles)/min(len(shingles),len(ref_shingles))>=threshold:
                return rid
        return None

def build_experiment(data_root,config_path):
    root=Path(data_root)
    read=lambda name:[json.loads(s) for s in (root/name).read_text(encoding='utf8').splitlines() if s.strip()]
    arabic=read('arabic.jsonl');latin=read('latin_synthetic.jsonl')
    all_rows=arabic+latin;lookup={r['id']:r for r in all_rows}
    if len(lookup)!=len(all_rows):raise ValueError('Canonical record IDs are not unique')
    if any(r.get('license')!='CC-BY-4.0' for r in all_rows):raise ValueError('Unapproved experiment source license')
    config=json.loads(Path(config_path).read_text(encoding='utf8'))
    ids=config['evaluation_ids']
    if len(ids)!=len(set(ids)) or set(ids)-set(lookup):raise ValueError('Evaluation IDs missing or duplicated')
    evaluation=[]
    for rid in ids:
        row=dict(lookup[rid])
        if row['script']=='latin' and row['split']!='validation':
            raise ValueError('Latin evaluation must preserve the original heldout document split')
        row.update(source_split=row['split'],source_split_group=row['split_group'],split='validation')
        if row['script']=='arabic':
            row.update(split_group='diagnostic_'+rid,independent_book_holdout_verified=False,
                       split_basis='image_text_fragment_holdout_not_verified_book_independent')
        evaluation.append(row)
    scripts=['arabic','latin']
    indices={s:ReferenceIndex([r for r in evaluation if r['script']==s]) for s in scripts}
    image_refs={r['image_sha256']:r['id'] for r in evaluation}
    train=[];excluded=[]
    for row in arabic+[r for r in latin if r['split']=='train']:
        overlap=image_refs.get(row['image_sha256']) or indices[row['script']].find(row['text'])
        if overlap:
            excluded.append({**row,'reserved_reference_id':overlap,'exclusion_reason':'reference_image_or_bidirectional_text_containment'})
        else:train.append(row)
    write_jsonl(root/'train.jsonl',train);write_jsonl(root/'evaluation.jsonl',evaluation);write_jsonl(root/'heldout_exclusions.jsonl',excluded)
    report=dict(version=2,training_records=len(train),train_by_script=dict(Counter(r['script'] for r in train)),
                evaluation_records=len(evaluation),eval_by_script=dict(Counter(r['script'] for r in evaluation)),
                arabic_long_evaluation_records=sum(r['script']=='arabic' and len(r['text'])>=1200 for r in evaluation),
                holdout_related_exclusions=len(excluded),source_book_independence_verified=False,real_latin_scan_pairs=0,
                overlap_rule='Exact images, NFC whitespace equality, bidirectional full-text containment for >=10words, or >=85%five-word-shingle containment relative to smaller record.',
                configuration_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                train_manifest_sha256=hashlib.sha256((root/'train.jsonl').read_bytes()).hexdigest(),
                supersedes='v1 diagnostic split contained cropped references in longer training paragraphs; use v2 manifests.')
    (root/'training_input_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',required=True);p.add_argument('--config',required=True)
    a=p.parse_args();print(json.dumps(build_experiment(a.data_root,a.config),indent=2))
