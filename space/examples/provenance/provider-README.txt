BALOCHI OCR GROUND TRUTH DATASET
=================================
Version   : 1.0
Language  : Balochi (ISO 639-3: bal)
Script    : Balochi Nastaliq
License   : CC BY 4.0
Records   : 10,374 image-text pairs
Institution: Balochi Academy — NLP Ecosystem (Thaheer Production)

DESCRIPTION
-----------
Human-verified OCR ground truth for Balochi script. Every image-text pair
was collected and verified by trained linguists. Images are grayscale PNG
at 300 DPI, auto-trimmed with 10px padding.

DATASET STRUCTURE
-----------------
balochi-ocr-dataset/
  metadata.csv        CSV  : all records (pandas, R, Excel compatible)
  metadata.jsonl      JSONL: all records (PyTorch, TF, HF compatible)
  dataset_info.json   JSON : machine-readable dataset card
  README.txt          TEXT : this file
  images/             DIR  : 10,374 PNG image files

CSV / JSONL FIELDS
------------------
  id          Unique record identifier
  file_name   Relative path to image (e.g. images/filename.png)
  text        Balochi ground truth transcription (UTF-8)
  char_count  Character count of transcription
  line_count  Number of text lines in image
  width       Image width in pixels (after processing)
  height      Image height in pixels (after processing)

LOADING EXAMPLES
----------------

Python / pandas:
  import pandas as pd
  df = pd.read_csv('balochi-ocr-dataset/metadata.csv')
  print(df.head())

PyTorch Dataset:
  from PIL import Image
  from torch.utils.data import Dataset
  import pandas as pd

  class BalochiOCR(Dataset):
      def __init__(self, root, transform=None):
          self.root = root
          self.meta = pd.read_csv(f'{root}/metadata.csv')
          self.transform = transform
      def __len__(self): return len(self.meta)
      def __getitem__(self, i):
          row = self.meta.iloc[i]
          img = Image.open(f'{self.root}/{row.file_name}')
          if self.transform: img = self.transform(img)
          return img, row.text

  dataset = BalochiOCR('balochi-ocr-dataset', transform=transforms.ToTensor())

TensorFlow / Keras:
  import tensorflow as tf, pandas as pd
  ROOT = 'balochi-ocr-dataset'
  df = pd.read_csv(f'{ROOT}/metadata.csv')
  ds = tf.data.Dataset.from_tensor_slices((df.file_name.tolist(), df.text.tolist()))
  ds = ds.map(lambda p, t: (
      tf.io.decode_png(tf.io.read_file(ROOT + '/' + p), channels=1), t
  ))

HuggingFace datasets:
  from datasets import load_dataset
  ds = load_dataset('imagefolder', data_dir='balochi-ocr-dataset')

PaddleOCR training data:
  import pandas as pd
  df = pd.read_csv('balochi-ocr-dataset/metadata.csv')
  with open('train_list.txt', 'w', encoding='utf-8') as f:
      for _, r in df.iterrows():
          f.write(f"{r.file_name}\t{r.text}\n")

CITATION
--------
  Balochi Academy NLP Ecosystem, 2026.
  Balochi OCR Ground Truth Dataset v1.0.
  Thaheer Production — https://balochiacademy.com
