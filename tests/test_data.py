import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_repeated_filename_uses_occurrence():
    from balocr.data import resolve_member
    assert resolve_member('images/a.png', 0) == 'images/a.png'
    assert resolve_member('images/a.png', 2) == 'images/a__dup2.png'
    assert resolve_member('images/a..png', 1) == 'images/a.__dup1.png'

@pytest.mark.parametrize('name', ['../x', '/x', 'C:/x', 'a/../../b', r'a\..\x'])
def test_archive_paths_are_confined(name):
    from balocr.data import validate_member
    with pytest.raises(ValueError):
        validate_member(name)

def test_group_split_never_separates_duplicates():
    from balocr.data import assign_splits
    rows = [dict(id='a', image_sha256='1', text='ā č', capture_date='2026-03-01'),
            dict(id='b', image_sha256='2', text='ā č', capture_date='2026-03-02'),
            dict(id='c', image_sha256='2', text='other', capture_date=None)]
    result = assign_splits(rows)
    assert len({r['split'] for r in result}) == 1
    assert len({r['split_group'] for r in result}) == 1

def test_conflicting_image_labels_are_quarantined():
    from balocr.data import partition_pairs
    good, bad = partition_pairs([
        dict(id='a', image_sha256='1', text='ā'),
        dict(id='b', image_sha256='1', text='a'),
        dict(id='c', image_sha256='2', text='č')])
    assert [r['id'] for r in good] == ['c']
    assert {r['id'] for r in bad} == {'a', 'b'}

def test_preparation_rejects_wrong_dimensions_and_moves_conflicts(tmp_path):
    import io,json,zipfile
    from PIL import Image
    from balocr.data import prepare_arabic
    buffer=io.BytesIO();Image.new('L',(40,30),255).save(buffer,format='PNG')
    rows=[dict(id='a',file_name='images/a.png',text='ā',width=40,height=30),
          dict(id='b',file_name='images/a.png',text='č',width=40,height=30),
          dict(id='c',file_name='images/c.png',text='x',width=99,height=30)]
    archive=tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w') as z:
        prefix='balochi-ocr-dataset/'
        z.writestr(prefix+'dataset_info.json',json.dumps({'license':'cc-by-4.0'}))
        z.writestr(prefix+'README.txt','CC BY 4.0')
        z.writestr(prefix+'metadata.csv','')
        z.writestr(prefix+'metadata.jsonl','\n'.join(json.dumps(r) for r in rows))
        for name in ['images/a.png','images/a__dup1.png','images/c.png']:
            z.writestr(prefix+name,buffer.getvalue())
    output=tmp_path/'prepared'
    assert prepare_arabic(archive,output)==[]
    quarantined=[json.loads(s) for s in (output/'quarantine.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(quarantined)==3
    for r in quarantined:
        if r['id'] in ('a','b'):
            assert r['image'].startswith('quarantine_images/')
            assert (output/r['image']).exists()
