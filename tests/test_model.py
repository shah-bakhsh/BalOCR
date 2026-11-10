import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_assistant_only_labels_and_prefix_validation():
    from balocr.model import assistant_labels
    assert assistant_labels([1,2,3,4], [1,2]) == [-100,-100,3,4]
    with pytest.raises(ValueError, match='prefix'):
        assistant_labels([1,9,3], [1,2])

def test_sequence_limit_rejects_whole_example():
    from balocr.model import assistant_labels
    with pytest.raises(ValueError, match='length'):
        assistant_labels([1,2,3,4], [1,2], max_length=3)

def test_diacritic_errors_are_counted():
    from balocr.metrics import score_predictions
    refs=[dict(id='a',text='ā č',script='latin',synthetic=True)]
    result=score_predictions(refs,[dict(id='a',text='a č',finished=True)])
    assert result['all']['character_edits'] == 1
    assert result['all']['reference_characters'] == 3
    assert result['all']['word_edits'] == 1
    assert result['all']['reference_words'] == 2

def test_missing_predictions_cannot_pass_evaluation():
    from balocr.metrics import score_predictions
    with pytest.raises(ValueError, match='coverage'):
        score_predictions([dict(id='a',text='x',script='arabic',synthetic=False)],[])

def test_capped_prediction_is_scored_and_flagged():
    from balocr.metrics import score_predictions
    result=score_predictions([dict(id='a',text='abcd',script='arabic',synthetic=False)],
                             [dict(id='a',text='a',finished=False)])
    assert result['all']['character_edits'] == 3
    assert result['all']['unfinished_predictions'] == 1

def test_nfc_metric_keeps_diacritics():
    from balocr.metrics import score_predictions
    result=score_predictions([dict(id='a',text='a\u0304',script='latin',synthetic=True)],
                             [dict(id='a',text='ā',finished=True)])
    assert result['all']['character_edits'] > 0
    assert result['all']['nfc_whitespace_character_edits'] == 0

def test_manifest_predictions_feed_metrics_directly(tmp_path, monkeypatch):
    import json
    from PIL import Image
    import balocr.model as model
    from balocr.metrics import score_predictions
    Image.new('RGB',(40,40),'white').save(tmp_path/'a.png')
    refs=[dict(id='a',image='a.png',text='ā č',script='latin',synthetic=True)]
    monkeypatch.setattr(model,'predict',lambda *a,**kw:dict(text='ā č',finished=True))
    output=tmp_path/'predictions.jsonl'
    model.predict_manifest(None,None,refs,tmp_path,output)
    predictions=[json.loads(line) for line in output.read_text(encoding='utf8').splitlines()]
    assert score_predictions(refs,predictions)['all']['cer']==0
