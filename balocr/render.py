"""Exact-label synthetic printed Latin images using HarfBuzz and FreeType."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re

import freetype
from fontTools.ttLib import TTFont
import uharfbuzz as hb
from PIL import Image

from .data import sha, text_key, write_jsonl

def text_spans(text, max_chars=220):
    words = list(re.finditer(r'\S+', text))
    result, start = [], 0
    while start < len(words):
        end = start+1
        while end < len(words) and words[end].end()-words[start].start() <= max_chars:
            end += 1
        a, b = words[start].start(), words[end-1].end()
        original = text[a:b]
        result.append(dict(text=' '.join(original.split()), text_original_span=original, start=a, end=b))
        start = end
    return result

@lru_cache(maxsize=8)
def font_data(path):
    data = Path(path).read_bytes()
    with TTFont(path) as tt:
        cmap = set(tt.getBestCmap())
    return data, cmap

def render_text(text, font_path, size=28, max_width=1000, padding=20):
    font_path = str(Path(font_path).resolve())
    data, cmap = font_data(font_path)
    missing = sorted({ord(c) for c in text if not c.isspace() and ord(c) not in cmap})
    if missing:
        raise ValueError('Missing font glyphs: '+','.join(f'U+{c:04X}' for c in missing))
    face = freetype.Face(font_path)
    face.set_pixel_sizes(0, size)
    font = hb.Font(hb.Face(data))
    font.scale = (size*64, size*64)
    hb.ot_font_set_funcs(font)
    def shape(line):
        buf = hb.Buffer()
        buf.add_str(line)
        buf.guess_segment_properties()
        hb.shape(font, buf)
        if any(g.codepoint == 0 for g in buf.glyph_infos):
            raise ValueError('Shaper emitted missing glyph')
        return list(zip(buf.glyph_infos, buf.glyph_positions))
    lines, line = [], ''
    for word in text.split(' '):
        candidate = (line+' '+word).lstrip()
        width = sum(pos.x_advance for _,pos in shape(candidate))/64
        if line and width > max_width-2*padding:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    if not lines:
        raise ValueError('Empty render text')
    masks = []
    line_height = max(size*1.5, face.size.height/64+6)
    for n, line in enumerate(lines):
        x = 0
        baseline = n*line_height
        for glyph, pos in shape(line):
            face.load_glyph(glyph.codepoint, freetype.FT_LOAD_RENDER)
            slot = face.glyph
            bitmap = slot.bitmap
            if bitmap.width and bitmap.rows:
                mask = Image.frombytes('L', (bitmap.width, bitmap.rows), bytes(bitmap.buffer), 'raw', 'L', bitmap.pitch)
                masks.append((mask, round(x+pos.x_offset/64+slot.bitmap_left),
                              round(baseline-pos.y_offset/64-slot.bitmap_top)))
            x += pos.x_advance/64
    if not masks:
        raise ValueError('No visible glyphs')
    left = min(x for _,x,y in masks)
    top = min(y for _,x,y in masks)
    right = max(x+m.width for m,x,y in masks)
    bottom = max(y+m.height for m,x,y in masks)
    image = Image.new('L', (right-left+2*padding, bottom-top+2*padding), 255)
    for mask,x,y in masks:
        image.paste(0, (x-left+padding, y-top+padding), mask)
    return image

def latin_sources(root):
    result = []
    snapshot=Path(root)/'latin_source_documents.jsonl'
    if snapshot.exists():
        result=[{**json.loads(line),'script':'latin'} for line in snapshot.read_text(encoding='utf8').splitlines() if line.strip()]
    for split in ('train', 'validation', 'test'):
        p = Path(root)/(split+'.jsonl')
        if not p.exists():
            continue
        for line in p.read_text(encoding='utf8').splitlines():
            row = json.loads(line)
            if row.get('script') != 'latin':
                continue
            if row.get('source_license') != 'CC-BY-4.0':
                raise ValueError('Unapproved Latin source license')
            result.append({**row, 'source_split': split})
    groups = {}
    for r in result:
        if r.get('source_license')!='CC-BY-4.0':
            raise ValueError('Unapproved Latin source license')
        group = r['final_split_group_id']
        if group in groups and groups[group] != r['source_split']:
            raise ValueError('Original Latin source group crosses splits')
        groups[group] = r['source_split']
    return result

def prepare_latin(root, fonts, output):
    output = Path(output)
    images = output/'images'/'latin'
    images.mkdir(parents=True, exist_ok=True)
    sources = latin_sources(root)
    # Only the fields needed to reproduce the exact text and attribution.
    fields = ('id','text_clean','source_id','source_url','source_license','license_evidence_url',
              'document_id','final_split_group_id','dialect','orthography','source_split',
              'attribution','original_author','title')
    write_jsonl(output/'source_metadata'/'latin_source_documents.jsonl',
                [{k:r.get(k) for k in fields} for r in sources])
    spans = [(r,s) for r in sources for s in text_spans(r['text_clean'])]
    protected = {text_key(s['text']) for r,s in spans if r['source_split'] != 'train'}
    result, excluded = [], []
    for r, span in spans:
        key = text_key(span['text'])
        if r['source_split']=='train' and key in protected:
            excluded.append(dict(source_id=r['id'], **span, reason='exact_span_in_protected_split'))
            continue
        base_id = 'latin_'+sha(f"{r['id']}:{span['start']}:{span['end']}".encode())[:22]
        for n, font_path in enumerate(fonts):
            image_id = f'{base_id}_{n}'
            size = (24,30,36)[int(sha(image_id.encode())[:4],16)%3]
            try:
                image = render_text(span['text'], font_path, size)
            except ValueError as exc:
                excluded.append(dict(source_id=r['id'], **span, font=font_path.name, reason=str(exc)))
                continue
            path = images/(image_id+'.png')
            image.save(path)
            raw = path.read_bytes()
            result.append(dict(id=image_id, text=span['text'], text_original_span=span['text_original_span'],
                               source_span=[span['start'],span['end']],
                               source_document_id=r['id'], source_text_sha256=sha(r['text_clean'].encode()),
                               source_id=r['source_id'], source_url=r['source_url'],
                               license=r['source_license'], license_evidence_url=r['license_evidence_url'],
                               attribution_reference='source_metadata/latin_source_documents.jsonl',
                               script='latin', dialect=r.get('dialect'), orthography=r.get('orthography'),
                               synthetic=True, synthetic_method='HarfBuzz_FreeType_printed_text',
                               normalization_operations=['collapse_whitespace_for_rendering'],
                               font=font_path.name, font_sha256=sha(font_path.read_bytes()), font_pixels=size,
                               split=r['source_split'], split_group=r['final_split_group_id'],
                               base_text_example_id=base_id, project_native_reviewed=False,
                               image=path.relative_to(output).as_posix(), width=image.width, height=image.height,
                               image_sha256=sha(str(image.size).encode()+image.tobytes()),
                               png_sha256=sha(raw), text_sha256=sha(span['text'].encode())))
    write_jsonl(output/'latin_synthetic.jsonl',result)
    write_jsonl(output/'latin_excluded.jsonl',excluded)
    report = dict(source_documents=len(sources), source_whitespace_words=sum(len(r['text_clean'].split()) for r in sources),
                  synthetic_images=len(result), unique_text_spans=len({r['base_text_example_id'] for r in result}),
                  split_counts=dict(Counter(r['split'] for r in result)),
                  exclusions=len(excluded), real_latin_scan_pairs=0, project_native_reviewed_pairs=0,
                  fonts=[p.name for p in fonts],
                  limitation='Synthetic printed images only; no claim of real scans, handwriting accuracy or new authored language data.')
    (output/'latin_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
    return result

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--fonts', required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    prepare_latin(args.source, sorted(Path(args.fonts).glob('*.ttf')), args.output)
