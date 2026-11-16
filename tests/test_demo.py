from PIL import Image
import pytest

from balocr.demo import prepare_image, format_result, recognize


def test_missing_image_rejected_before_model_loading():
    with pytest.raises(ValueError, match='Upload'):
        prepare_image(None)


def test_large_image_rejected_without_resizing_text():
    image = Image.new('1', (5001, 5000))
    with pytest.raises(ValueError, match='25 megapixels'):
        prepare_image(image)


def test_transparent_background_becomes_white():
    image = Image.new('RGBA', (2, 1), (0, 0, 0, 0))
    image.putpixel((1, 0), (0, 0, 0, 255))
    result = prepare_image(image)
    assert result.mode == 'RGB'
    assert result.getpixel((0, 0)) == (255, 255, 255)
    assert result.getpixel((1, 0)) == (0, 0, 0)


def test_photo_orientation_is_applied_before_ocr():
    image = Image.new('RGB', (3, 2), 'white')
    image.getexif()[274] = 6
    assert prepare_image(image).size == (2, 3)


def test_unfinished_result_preserves_unicode_and_flags_the_cap():
    text = 'بلوچی ā č ẽ\n'
    output, status = format_result({'text': text, 'finished': False,
                                   'generated_tokens': 512}, 12.34)
    assert output == text
    assert 'incomplete' in status.lower()
    assert '512' in status and '12.3' in status


def test_finished_result_is_not_reported_as_incomplete():
    output, status = format_result({'text': 'بلوچی', 'finished': True,
                                   'generated_tokens': 5}, 2.0)
    assert output == 'بلوچی'
    assert 'incomplete' not in status.lower()
    assert 'accuracy' not in status.lower()


@pytest.mark.parametrize('tokens', [0, 2049, 128.5, None, 'bad'])
def test_invalid_token_limit_rejected_before_model_loading(tokens):
    with pytest.raises(ValueError, match='128 to 2048'):
        recognize(Image.new('RGB', (2, 2)), tokens)


def test_composed_space_keeps_upload_cleanup_and_both_apis():
    # Characterize the outer Blocks: TabbedInterface drops child cleanup.
    gr = pytest.importorskip('gradio')
    from space.shared_app import compose_demo
    with gr.Blocks(delete_cache=(600, 600)) as ocr:
        output = gr.Textbox()
        gr.Button().click(lambda: 'image text', outputs=output, api_name='ocr')
    with gr.Blocks() as chat:
        output = gr.Textbox()
        gr.Button().click(lambda: 'continuation', outputs=output, api_name='complete')
    demo = compose_demo(ocr, chat)
    assert demo.delete_cache == (600, 600)
    names = {d['api_name'] for d in demo.config['dependencies']}
    assert {'ocr', 'complete'} <= names
