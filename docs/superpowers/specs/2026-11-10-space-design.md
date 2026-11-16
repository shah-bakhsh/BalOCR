# BalOCR testing Space

The user wants a Hugging Face Space to test BalOCR by uploading Balochi images. Existing authorization covers public commercial-data demos and code, with existing resources only. Full BalOCR training is incomplete; no final adapter exists. The demo must identify its actual model as the pinned pretrained PaddleOCR-VL-1.6, not imply completed BalOCR fine-tuning.

The initial target was a new public `shah-bakhsh/BalOCR` Gradio Space on CPU Basic. Hugging Face rejected creation with HTTP402: a new compute Space requires PRO. The existing private ocr-test Space actually hosts BalParser and must remain untouched. Add BalOCR as a CPU-only tab in the existing public `shah-bakhsh/balcorpus-balochi-demo` Space, preserving the original BalChat app files and `/complete` API. Keep its existing ZeroGPU hardware request unchanged; no new paid hardware, subscription or provider API is requested. Retain the previous README/requirements and immutable commit. The standalone CPU recipe remains available for a future dedicated Space.

Reuse `balocr.model.load_model` and `predict` with native Transformers, FP32/eager attention and the frozen revision. Load the model lazily once; serialize model execution and queue at most eight requests. A server-backed demo would depend on currently broken CUDA; a pilot-adapter demo would knowingly use an unselected model with worse Arabic diagnostics. CPU baseline inference is the useful available path.

Accept one printed page/crop image, honor EXIF orientation, composite transparency on white and reject oversized images. Offer a bounded output length. Return visible text unchanged plus timing and an explicit incomplete-output flag when EOS is absent. Display Arabic and Latin naturally without translation/transliteration. Ship one attributed Arabic example and one synthetic Latin example from the approved diagnostic corpus; describe their source licenses and limitations.

No uploaded documents or predictions are added to the dataset or written to project training logs. Gradio temporary uploads are deleted on a short cache schedule. The hosting provider processes uploads. No native certification, handwriting accuracy or production accuracy is claimed.

Verify validation/formatting behavior with lightweight tests, then exercise the actual deployed upload-to-OCR endpoint on both example images. Observe build/runtime errors and fix them before delivering the link. Public source and receipts must identify exact deployed files/model revision/hardware and any unresolved limitations.
