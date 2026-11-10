"""Strict OCR evaluation with no missing/unfinished prediction filtering."""
from collections import defaultdict
import argparse
import json
import unicodedata
from pathlib import Path

def distance(a,b):
    if len(a)<len(b):a,b=b,a
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        now=[i]
        for j,y in enumerate(b,1):
            now.append(min(now[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=now
    return prev[-1]

def normalize(text):
    return ' '.join(unicodedata.normalize('NFC',text).split())

def score_predictions(references,predictions):
    refs={r['id']:r for r in references}
    preds={r['id']:r for r in predictions}
    if len(refs)!=len(references) or len(preds)!=len(predictions) or set(refs)!=set(preds):
        raise ValueError('Prediction coverage/unique IDs must exactly match reference IDs')
    if not refs:raise ValueError('Cannot evaluate an empty set')
    groups=defaultdict(list)
    for rid,r in refs.items():
        groups['all'].append((r,preds[rid]))
        groups[r['script']+('_synthetic' if r['synthetic'] else '_real')].append((r,preds[rid]))
    result={}
    for name,pairs in groups.items():
        stats=dict(examples=len(pairs),unfinished_predictions=sum(not p.get('finished',False) for r,p in pairs),
                   character_edits=0,reference_characters=0,word_edits=0,reference_words=0,
                   nfc_whitespace_character_edits=0,nfc_whitespace_reference_characters=0)
        for r,p in pairs:
            a,b=r['text'],p['text']
            if not a:raise ValueError('Empty reference label')
            stats['character_edits']+=distance(a,b)
            stats['reference_characters']+=len(a)
            stats['word_edits']+=distance(a.split(),b.split())
            stats['reference_words']+=len(a.split())
            a,b=normalize(a),normalize(b)
            stats['nfc_whitespace_character_edits']+=distance(a,b)
            stats['nfc_whitespace_reference_characters']+=len(a)
        stats['cer']=stats['character_edits']/stats['reference_characters']
        stats['wer']=stats['word_edits']/stats['reference_words'] if stats['reference_words'] else None
        stats['nfc_whitespace_cer']=stats['nfc_whitespace_character_edits']/stats['nfc_whitespace_reference_characters'] if stats['nfc_whitespace_reference_characters'] else None
        result[name]=stats
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--references',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    read=lambda p:[json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines() if s.strip()]
    result=score_predictions(read(a.references),read(a.predictions))
    Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
