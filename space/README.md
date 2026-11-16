---
title: BalOCR — Balochi OCR testing
emoji: 📖
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
models:
- PaddlePaddle/PaddleOCR-VL-1.6
datasets:
- shah-bakhsh/BalDataset
tags:
- balochi
- ocr
- image-to-text
startup_duration_timeout: 30m
preload_from_hub:
- PaddlePaddle/PaddleOCR-VL-1.6 model.safetensors,config.json,generation_config.json,preprocessor_config.json,processor_config.json,tokenizer.json,tokenizer.model,tokenizer_config.json,special_tokens_map.json,added_tokens.json,chat_template.jinja 132c45f52c0a26ed5a6b2f6bd8e3985794cc27dd
---

# BalOCR testing Space

Upload a printed Balochi image in Arabic or Latin script and extract its visible text. A small crop is recommended on this CPU demo. The output length is bounded, and incomplete output is flagged.

**Actual model:** the original pretrained `PaddlePaddle/PaddleOCR-VL-1.6`, pinned to `132c45f52c0a26ed5a6b2f6bd8e3985794cc27dd`, with native Transformers code (`trust_remote_code=False`), FP32 and eager attention. **There is no completed BalOCR fine-tuned adapter in this demo.** Full training remains unfinished because the available training server cannot initialize CUDA. No perfect accuracy, native-speaker certification, or handwriting benchmark is claimed.

Hardware: CPU Basic; no paid GPU or provider endpoint. The first OCR request downloads/loads the pinned model; requests are serialized. CPU inference may take several minutes for a long crop. The example Latin image is synthetic printed text, not a real scan. See `examples/ATTRIBUTION.md` for the CC BY 4.0 image/text credits and OFL font provenance.

The app does not add uploads or predictions to the training dataset. Gradio temporary uploads are scheduled for deletion after about 10 minutes; Hugging Face hosts and processes them.

Project code: MIT. Base model: Apache 2.0. Example source text/images: CC BY 4.0. Fonts used to render Latin examples: SIL OFL 1.1. Individual attribution and modification notes accompany the examples.

[Source code and training status](https://github.com/shah-bakhsh/BalOCR) · [Complete prepared data](https://huggingface.co/datasets/shah-bakhsh/BalDataset/tree/main/ocr/canonical)
