import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import pytest

def helper():
    path=Path(__file__).resolve().parents[1]/'tools/download_ocr_release.py'
    spec=importlib.util.spec_from_file_location('balocr_data_download',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_download_rejects_archive_traversal(tmp_path):
    module=helper();archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../outside.txt','unsafe')
    with pytest.raises(ValueError,match='Unsafe'):module.extract_checked(archive,tmp_path/'output')
    assert not (tmp_path/'outside.txt').exists()

@pytest.mark.parametrize('include_provenance',[False,True])
def test_download_requires_attribution_snapshot(tmp_path,monkeypatch,include_provenance):
    module=helper();files=tmp_path/'files';files.mkdir()
    base=files/'base.zip'
    with zipfile.ZipFile(base,'w') as z:z.writestr('old.txt','original')
    overlay=files/'overlay.zip';text='ā č'
    with zipfile.ZipFile(overlay,'w') as z:
        z.writestr('training_input_report.json',json.dumps({'training_records':1,'evaluation_records':0}))
        z.writestr('train.jsonl',json.dumps({'id':'a','text':text,'attribution_reference':'source_metadata/source.jsonl'})+'\n')
        z.writestr('evaluation.jsonl','')
        if include_provenance:z.writestr('source_metadata/source.jsonl','[]')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'training_records':1,'base_revision':'b'*40,'base_manifest_path':'base.json',
              'base_archive_sha256':digest(base),'overlay_path':'overlay.zip','overlay_sha256':digest(overlay),
              'provenance_files':[{'path':'source_metadata/source.jsonl','sha256':hashlib.sha256(b'[]').hexdigest()}]}
    base_manifest={'parts':[{'path':'base.zip','sha256':digest(base)}]}
    (files/'release.json').write_text(json.dumps(manifest));(files/'base.json').write_text(json.dumps(base_manifest))
    monkeypatch.setattr(module,'HfApi',lambda:SimpleNamespace(repo_info=lambda *a,**kw:SimpleNamespace(sha='a'*40)))
    monkeypatch.setattr(module,'hf_hub_download',lambda repo,name,**kw:str(files/('release.json' if name=='ocr/v2/release.json' else name)))
    monkeypatch.setattr(sys,'argv',['download','--output',str(tmp_path/'result')])
    if include_provenance:
        module.main()
        assert (tmp_path/'result/prepared/source_metadata/source.jsonl').read_text()=='[]'
    else:
        with pytest.raises(ValueError,match='attribution'):module.main()

def test_complete_download_extracts_canonical_data(tmp_path,monkeypatch):
    module=helper();archive=tmp_path/'canonical.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('data/prepared/training_input_report.json',json.dumps({'training_records':1,'evaluation_records':0}))
        z.writestr('data/prepared/train.jsonl',json.dumps({'id':'a','text':'ā č','attribution_reference':'source_metadata/source.jsonl'})+'\n')
        z.writestr('data/prepared/evaluation.jsonl','')
        z.writestr('data/prepared/source_metadata/source.jsonl','[]')
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest={'training_records':1,'evaluation_records':0,'archive_sha256':sha,
              'parts':[{'path':'canonical.part000','sha256':sha,'bytes':archive.stat().st_size}]}
    mf=tmp_path/'manifest.json';mf.write_text(json.dumps(manifest))
    monkeypatch.setattr(module,'HfApi',lambda:SimpleNamespace(repo_info=lambda *a,**kw:SimpleNamespace(sha='a'*40)))
    monkeypatch.setattr(module,'hf_hub_download',lambda repo,name,**kw:str(mf if name.endswith('release.json') else archive))
    monkeypatch.setattr(sys,'argv',['download','--complete','--output',str(tmp_path/'result')])
    module.main()
    assert (tmp_path/'result/data/prepared/source_metadata/source.jsonl').read_text()=='[]'
