"""Public BalOCR testing interface; deployed together with the balocr SDK."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
import gradio as gr

from balocr.demo import recognize

ROOT = Path(__file__).resolve().parent
SPACE_REPO = os.environ.get('SPACE_ID', 'shah-bakhsh/BalOCR')
EXAMPLE_PREFIX = 'balocr_demo/' if SPACE_REPO == 'shah-bakhsh/balcorpus-balochi-demo' else ''


def extract_text(image, max_tokens):
    try:
        return recognize(image, max_tokens)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        # Do not include uploaded content or internal paths in public errors.
        print(f'OCR request failed: {type(exc).__name__}', flush=True)
        raise gr.Error('OCR could not finish. Try a smaller, clearer crop or retry shortly.') from None


with gr.Blocks(title='BalOCR — Balochi OCR testing', delete_cache=(600, 600)) as demo:
    gr.Markdown('''# BalOCR
### Read Balochi text from an image
Upload printed Arabic- or Latin-script text, then select **Extract text**.

**Testing model: pretrained PaddleOCR-VL-1.6.** BalOCR fine-tuning is unfinished; this demo currently uses the original OCR model. Balochi accuracy has not been certified.
''')
    with gr.Row():
        with gr.Column():
            image = gr.Image(label='Printed page or text crop', type='pil', image_mode='RGBA',
                             sources=['upload', 'clipboard'], format='png')
            length = gr.Slider(128, 2048, value=768, step=128, label='Maximum output length',
                               info='Increase for longer text. Small crops work best on this CPU demo.')
            run = gr.Button('Extract text', variant='primary')
            gr.Examples([[str(ROOT/'examples/arabic.png')], [str(ROOT/'examples/latin-synthetic.png')]],
                        inputs=[image], label='Try an example (Arabic / synthetic Latin)', cache_examples=False)
        with gr.Column():
            text = gr.Textbox(label='Extracted text', lines=16, buttons=['copy'], elem_classes=['ocr-output'])
            status = gr.Markdown('CPU inference can take a few minutes. The first request also loads the model.')
    run.click(extract_text, inputs=[image, length], outputs=[text, status], api_name='ocr',
              concurrency_limit=1, concurrency_id='ocr-model')
    gr.ClearButton([image, text, status])
    gr.Markdown(f'''The model transcribes visible text; it does not translate or change the script. Use a clear crop for small print. Output can contain errors; compare it with the image.

Uploads are processed on this Hugging Face Space and are not added to our training dataset. Temporary uploads are scheduled for deletion after about 10 minutes.

[Dataset and attribution](https://huggingface.co/datasets/shah-bakhsh/BalDataset/tree/main/ocr/canonical) · [Code and training status](https://github.com/shah-bakhsh/BalOCR) · [Example credits](https://huggingface.co/spaces/{SPACE_REPO}/blob/main/{EXAMPLE_PREFIX}examples/ATTRIBUTION.md)
''')

demo.queue(max_size=8, default_concurrency_limit=1)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', max_file_size='20mb', show_error=False,
                css='.ocr-output textarea {unicode-bidi: plaintext; text-align: start;}')
