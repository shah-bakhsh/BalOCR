import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FONTS = Path(__file__).resolve().parents[1]/'assets'/'fonts'

def test_source_spans_preserve_diacritics():
    from balocr.render import text_spans
    text = 'ā ē ī ō ū č š ž ṭ ḍ ṛ ǰ ẽ ẫ ễ ȭ ġ ɣ γ a\u0304'
    spans = text_spans(text, max_chars=18)
    assert ' '.join(s['text'] for s in spans) == text
    assert all(text[s['start']:s['end']] == s['text_original_span'] for s in spans)

def test_render_has_margins_and_keeps_combining_marks():
    from balocr.render import render_text
    image = render_text('Balōčī ā š č ẽ ɣ γ a\u0304', FONTS/'NotoSans.ttf', 30)
    assert image.mode == 'L'
    from PIL import ImageChops, Image
    box = ImageChops.difference(image, Image.new('L', image.size, 255)).getbbox()
    assert box and box[0] >= 12 and box[1] >= 12
    assert image.width-box[2] >= 12 and image.height-box[3] >= 12

def test_missing_glyph_rejected():
    from balocr.render import render_text
    with pytest.raises(ValueError, match='glyph'):
        render_text('text \U0001f680', FONTS/'NotoSans.ttf', 24)

def test_latin_splits_inherit_documents(tmp_path):
    import json
    from balocr.render import latin_sources
    root = tmp_path
    for split in ['train','validation']:
        row=dict(id=split,text_clean='Balōčī',script='latin',source_license='CC-BY-4.0',final_split_group_id=split+'_document')
        (root/(split+'.jsonl')).write_text(json.dumps(row)+'\n',encoding='utf8')
    rows = latin_sources(root)
    assert sum(r['source_split']=='validation' for r in rows) == 1
    assert all(r['source_license']=='CC-BY-4.0' for r in rows)
    groups = {}
    for r in rows:
        groups.setdefault(r['final_split_group_id'], set()).add(r['source_split'])
    assert all(len(s)==1 for s in groups.values())

def test_snapshot_sources_regenerate_without_sibling_project(tmp_path):
    import json
    from balocr.render import latin_sources
    row=dict(id='document',text_clean='ā č',source_license='CC-BY-4.0',final_split_group_id='group',source_split='validation')
    (tmp_path/'latin_source_documents.jsonl').write_text(json.dumps(row)+'\n',encoding='utf8')
    result=latin_sources(tmp_path)
    assert len(result)==1 and result[0]['text_clean']=='ā č'
    assert result[0]['source_split']=='validation'
