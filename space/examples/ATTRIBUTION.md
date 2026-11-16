# Example attribution

Both examples are selected from the frozen BalOCR v2 diagnostic corpus, not generated Balochi language. Their exact metadata and references are in `records.json`. This public testing demo reveals the examples, so future independent evaluations should account for that exposure.

## Arabic printed image

`arabic.png`: source record `vbdJlOIJ1xc9U2NOUcXa`. Balochi Academy NLP Ecosystem / Thaheer Production, 2026. Source declares **CC BY 4.0**: https://creativecommons.org/licenses/by/4.0/ . The provider README and dataset_info evidence are retained in `provenance/`. The PNG is byte-identical to the retained canonical image; the release repair renamed its path. Original book identity is not available. Provider human-review claims are not independent project certification.

## Synthetic Latin printed image

`latin-synthetic.png`: source record `latin_dc05834e552f37bb256b71_1`, Coastal Balochi scholarly transcription. Source: https://fd-repo.uni-bamberg.de/records/gyws0-g4218 ; **CC BY 4.0**. Attribution from the frozen source metadata: null. Full citation and selected source document metadata/text are retained in `provenance/`.

Modification: an exact text span was rendered in Noto Serif with HarfBuzz/FreeType, collapsing source whitespace for printed layout while preserving diacritics and spelling. This is **synthetic**, not a real Latin scan. The rendered PNG is unchanged from the frozen corpus. Noto Serif was distributed under SIL OFL 1.1; its license and font source/hash ledger accompany this example. No font binary is required to view the image.

Dataset: https://huggingface.co/datasets/shah-bakhsh/BalDataset/tree/main/ocr/canonical . Project code is MIT; that does not replace the example-data or font licenses.
