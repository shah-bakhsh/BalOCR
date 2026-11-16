"""Add BalOCR without replacing the existing BalChat app or its API."""
import importlib.util
import os
from pathlib import Path
import sys

os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['GRADIO_ANALYTICS_ENABLED']='False'

import gradio as gr

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))

def load_app(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.demo

def compose_demo(ocr,chat):
    with gr.Blocks(title='Balochi model testing', analytics_enabled=False,
                   delete_cache=(600, 600)) as demo:
        with gr.Tabs():
            with gr.Tab('BalOCR'):
                ocr.render()
            with gr.Tab('BalChat'):
                chat.render()
    return demo

if __name__=='__main__':
    ocr=load_app('balocr_testing_ui',ROOT/'app.py')
    chat=load_app('existing_balchat_ui',ROOT.parent/'app.py')
    demo=compose_demo(ocr,chat)
    demo.queue(max_size=8,default_concurrency_limit=1).launch(
        server_name='0.0.0.0',max_file_size='20mb',show_error=False,
        css='.ocr-output textarea {unicode-bidi: plaintext; text-align: start;}')
