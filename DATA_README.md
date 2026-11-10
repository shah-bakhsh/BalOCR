# BalOCR complete paired data v2

This package preserves **11,195 canonical usable pairs**:10,265 Arabic-script pairs and930synthetic printed Latin images. The latter are465source text spans rendered in two frozen OFL fonts, from11licensed original documents. Synthetic variants are not new human-authored language data.

The original Arabic archive had10,374label records and10,372available PNGs. We repaired4,514renamed-image links using occurrence suffixes and exact recorded dimensions. Two missing images,46duplicate pairs and61conflicting image/label records are excluded from training. Original source metadata and the107available excluded images remain in `data/prepared/source_metadata/` and `quarantine_images/`.

The corrected **v2 experiment** uses11,043training examples(10,205Arabic and838synthetic Latin), plus48diagnostics(24Arabic, including12longer passages, and24synthetic Latin). Sixty Arabic records are excluded from training as reserved or overlapping text. The canonical Arabic source manifest remains train-only because original book IDs are absent. Canonical Latin has838train/92validation images, retaining the original protected document split.

V1's diagnostic split missed cropped-reference containment in longer training paragraphs. Use v2 `train.jsonl` and `evaluation.jsonl`. V2 checks found zero exact reference containment and zero >=85% smaller-record five-word-shingle containment under the documented >=10-word rule. This does not prove unseen-book independence. Project native-speaker reviews:zero. Real Latin scan evaluation pairs:zero. No perfect-accuracy or production-readiness claim is made.

Arabic retained labels contain1,893,658whitespace words. The Latin original documents contain17,268words, counted once before rendering. Preserve source spellings and diacritics; no transliteration or invented Balochi sentences were added.

Rights: paired source data/text declare CC BY4.0. See `SOURCES.json`, provider README/dataset_info snapshots and per-record source URLs. Font binaries and licenses are in `assets/fonts/` under SIL OFL1.1. Source presentation was changed to generate synthetic printed Latin examples, as explicitly recorded in metadata.

`DATA_MANIFEST.json` lists each packaged file's SHA256 and size. The ZIP has an external SHA256 sidecar and CRC audit. `config/experiment-v2.json` fixes the48diagnostic IDs; the source code reproduces the split using `python -m balocr.splits --data-root data/prepared --config config/experiment-v2.json`.

Code/SDK: https://github.com/shah-bakhsh/BalOCR . Public data: https://huggingface.co/datasets/shah-bakhsh/BalDataset/tree/main/ocr . These are data and preparation tools; a fully trained BalOCR model has not been certified by this release.
