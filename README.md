# BalOCR

Balochi printed-text OCR in Arabic and Latin scripts. This project prepares real image/transcription pairs, adds explicitly synthetic printed Latin images, and adapts a pretrained vision model. It is a research development release; it does not promise perfect OCR or native-speaker certification.

## Actual prepared data

| Data | Images/pairs | Notes |
|---|---:|---|
| Supplied Arabic archive | 10,374 label records; 10,372 PNGs | Source declares CC BY 4.0 and provider linguistic verification |
| Retained Arabic pairs | 10,265 | 4,514 filename links repaired using occurrence suffixes and matching dimensions |
| Excluded Arabic records | 109 | 2 missing images, 46 exact pair duplicates, 61 conflicting image/label records |
| Synthetic Latin images | 930 | 465 text spans, two OFL Noto fonts, 11 original licensed documents |
| Canonical Latin splits | 838 train; 92 validation | Original document groups preserved |
| Total retained image pairs | 11,195 | Includes synthetic variants; not 11,195 independent documents |

Arabic retained labels contain **1,893,658 whitespace words**. The Latin source documents contain **17,268 words**, counted once before rendering. Synthetic images add visual examples, not new authored Balochi language. The Latin sources include scholarly transcription, with diacritics preserved; they do not cover every community spelling convention.

See `reports/arabic_report.json`, `reports/latin_report.json`, `data/prepared/source_metadata/` and `SOURCES.json` for provenance. Archive metadata claims human review; **this project has performed zero native-speaker reviews**. Repaired filename associations are not independently native-reviewed.

## Model and hardware

Initial base: [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6), pinned revision `132c45f52c0a26ed5a6b2f6bd8e3985794cc27dd`, Apache 2.0. It is a compact OCR vision model, not a text-only chat model. Native Transformers code loads it with `trust_remote_code=False`, checks loading mismatches and preserves untied pretrained embeddings.

The official Paddle deployment requirements specify GPU compute capability >=7.0. The user's TITAN X Pascal cards are 6.1, so this project uses an **experimental, actually tested FP32/eager-attention path** with the existing Torch `2.14.1+cu126` build. It passed Arabic and Latin generation plus LoRA backward on a 12 GB card, with peak allocated GPU memory 4,304,699,904 bytes for the small preflight. That is not a guarantee that every long input fits. BF16, FlashAttention and quantized training are not enabled. Do not install a newer CUDA build that drops Pascal support.

## Setup

On Linux, starting from the existing Python installation with working CUDA PyTorch:

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -c constraints-pascal.txt
python -m pytest tests -q
```

For other hardware, install the appropriate PyTorch build separately and omit the Pascal constraint. CPU uses FP32 and needs several GB of free RAM; this laptop's low available RAM makes the Linux server the practical model host.

## Regenerate the data

Run from this project folder. `../BalCorpus-Final` is the existing frozen corpus, not a network dependency.

```powershell
python -m balocr.data --archive "C:/Users/Lenovo/Downloads/balochi-ocr-dataset.zip" --output data/prepared
python -m balocr.render --source ../BalCorpus-Final/data/recommended/commercial --fonts assets/fonts --output data/prepared
```

The data release includes the exact Latin source documents and font license snapshots. To regenerate Latin images from those accompanying snapshots, use `--source data/prepared/source_metadata`. Arabic raw labels, the source README and declared license are preserved. Missing/corrupt/conflicting pairs are quarantined rather than relabeled or supplied with invented text.

## Fine-tune

Use a manifest containing only `split=train` rows and keep evaluation IDs outside training. The derived full training manifest preserves the small diagnostic holdout. Maximum steps are explicit so a run cannot silently spend unlimited GPU time:

```bash
CUDA_VISIBLE_DEVICES=1 python -m balocr.train \
  --manifest data/prepared/train.jsonl --data-root data/prepared \
  --output runs/full --max-steps 2773 --accumulation 4 \
  --max-pixels 401408 --allow-experimental-pascal
```

Use the actual `ceil(full_training_records / 4)` from `training_input_report.json` for one nominal pass; the frozen full manifest has 11,090 records and therefore 2,773 nominal one-pass steps. Each run records the exact input hash, IDs actually seen, rejected over-length records, actual step count and loss history. Over-length labels are rejected whole, never silently truncated. Only the assistant transcription tokens contribute to loss. LoRA adapts decoder query/value projections with rank 8.

## Use the Python SDK

```python
from PIL import Image
from balocr.model import load_model, predict

model, processor = load_model(
    device="cuda:0", adapter="runs/full/adapter", allow_pascal=True
)
model.eval()
result = predict(model, processor, Image.open("page.png").convert("RGB"))
print(result["text"])
print(result["finished"])  # False means generation hit the token cap
```

Omit `adapter` to test the original pretrained model. That does not use a trained BalOCR adapter. OCR reproduces visible text; it does not automatically translate or convert the script.

## Test on a labeled manifest

```bash
python -m balocr.model --manifest data/prepared/evaluation.jsonl \
  --data-root data/prepared --adapter runs/full/adapter --device cuda:0 \
  --allow-experimental-pascal --output runs/full/predictions.jsonl
python -m balocr.metrics --references data/prepared/evaluation.jsonl \
  --predictions runs/full/predictions.jsonl --output runs/full/scores.json
```

Metrics require every reference ID exactly once, score unfinished predictions against the complete reference and report character/word error rates separately for real Arabic and synthetic Latin. Raw CER counts Unicode code points including diacritics; an additional NFC/whitespace-normalized CER also preserves diacritics. Lower is better.

## Evaluation limits

The archive does not identify original source books. Conservative date/duplicate grouping yielded **no defensible Arabic book-independent validation set**; canonical Arabic data is train-only. A separate 12-image Arabic diagnostic holdout is retained for the development experiment. It must not be described as a verified unseen-book benchmark. The Latin validation source is a held-out original document, but its images are synthetic. There are no real Latin scan evaluation pairs yet and no certified handwriting benchmark.

Before production use, obtain document-level provenance, real Latin scans with exact transcriptions, and native review of representative unseen Arabic/Latin documents. Character/word error rates on that benchmark should determine whether to ship. General OCR leaderboard scores do not measure Balochi accuracy.

## Rights

Source text and paired Arabic data: CC BY 4.0 declarations, with individual attribution in `SOURCES.json` and source metadata. Noto fonts: SIL OFL 1.1, in `assets/fonts/*-OFL.txt`. Base model: Apache 2.0. Project code: MIT. Synthetic rendering is explicitly documented as a modification of source text presentation.
