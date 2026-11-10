import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def test_full_reference_inside_long_training_paragraph_is_excluded():
    from balocr.splits import overlaps_reference
    reference=' '.join('word'+str(n) for n in range(25))
    train='Introduction '+reference+' additional long content '*40
    assert overlaps_reference(train,reference)

def test_training_crop_inside_reference_is_excluded():
    from balocr.splits import overlaps_reference
    crop=' '.join('word'+str(n) for n in range(15))
    assert overlaps_reference(crop,'Introduction '+crop+' and conclusion')

def test_diacritics_are_not_collapsed_by_leakage_check():
    from balocr.splits import overlaps_reference
    assert not overlaps_reference(' '.join(['a']*15),' '.join(['ā']*15))

def test_common_punctuation_is_not_a_document_overlap():
    from balocr.splits import overlaps_reference
    assert not overlaps_reference('۔','long reference with many words ۔ and other content')

def test_reordered_near_copy_is_excluded():
    from balocr.splits import overlaps_reference
    words=['word'+str(n) for n in range(70)]
    assert overlaps_reference(' '.join(words),' '.join(words[:32]+['inserted']+words[32:]))

def test_experiment_is_reproducible_and_keeps_original_manifests(tmp_path):
    import json
    from balocr.splits import build_experiment
    from balocr.data import write_jsonl
    text=' '.join('word'+str(n) for n in range(30))
    rows=[dict(id='short',text='different',script='arabic',split='train',split_group='unknown',image_sha256='a',license='CC-BY-4.0'),
          dict(id='crop',text=text,script='arabic',split='train',split_group='unknown',image_sha256='b',license='CC-BY-4.0'),
          dict(id='long',text='introduction '+text+' ending',script='arabic',split='train',split_group='unknown',image_sha256='c',license='CC-BY-4.0')]
    write_jsonl(tmp_path/'arabic.jsonl',rows);write_jsonl(tmp_path/'latin_synthetic.jsonl',[])
    original=(tmp_path/'arabic.jsonl').read_bytes()
    config=tmp_path/'config.json';config.write_text(json.dumps({'evaluation_ids':['crop']}))
    result=build_experiment(tmp_path,config)
    assert result['training_records']==1 and result['holdout_related_exclusions']==2
    assert (tmp_path/'arabic.jsonl').read_bytes()==original
    first=(tmp_path/'train.jsonl').read_bytes()
    build_experiment(tmp_path,config)
    assert (tmp_path/'train.jsonl').read_bytes()==first
