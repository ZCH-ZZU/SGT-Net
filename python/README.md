# Python code for SGT-Net

This folder contains the PyTorch implementation of the three SGT-Net stages:

| Stage | Entry point | Purpose |
| --- | --- | --- |
| Stage I | `main_Stage1.py` / `.ipynb` | intra-subject 5-fold GroupKFold |
| Stage II | `main_Stage2.py` / `.ipynb` | LOSO cross-subject adaptation + optional source selection |
| Stage III | `main_Stage3.py` / `.ipynb` | few-shot target personalization (DoRA/LoRA/full/linear probing) |

---

## 1. Requirements

```bash
cd python
pip install -r requirements.txt
```

---

## 2. Data

The code reads the HDF5 files produced by `../matlab/run_preprocessing.m`:

```
../datasets/hybridBCI_Tn8.h5
../datasets/stroke_patients_Tn8.h5
```

Each subject group contains:

```
eeg_data        [time, channels, samples]
hbo_data        [time, channels, samples]
hbr_data        [time, channels, samples]
labels_decimal  [1, samples]      (1-based)
trial_ids       [1, samples]      (1-based)
```

Trial-level splitting uses `trial_ids`: all windows of a trial share the same
`trial_ids` value.

---

## 3. Running

```bash
cd python

# Stage I: healthy / stroke
python main_Stage1.py --config config/Stage1_public.yaml
python main_Stage1.py --config config/Stage1_stroke.yaml

# Stage II: healthy / stroke
python main_Stage2.py --config config/Stage2_public.yaml
python main_Stage2.py --config config/Stage2_stroke.yaml

# Stage III: healthy / stroke
python main_Stage3.py --config config/Stage3_public.yaml
python main_Stage3.py --config config/Stage3_stroke.yaml
```

The notebooks `main_Stage1.ipynb`, `main_Stage2.ipynb`, and
`main_Stage3.ipynb` call the same functions. Run them from `python/`.

In notebooks, `logging.clear_output: true` clears the cell output after each
subject (Stage I) or target (Stage II/III), so the console does not accumulate.
Set `logging.clear_fold: true` in a Stage I config to also clear after each
fold. The flag is a no-op in a plain Python terminal.

Outputs are written under `results/Stage1_*`, `results/Stage2_*`, and
`results/Stage3_*`.

---

## 4. Configuration

`config/params.yaml` contains only the shared model parameters:

* two backbone variants: `no_dts` (w/o DTS) and `dts` (w/ DTS, default)
* five fusion types: `concat`, `co_attention`, `gated`, `lmf`, `tgca` (default)

`plotting.history.save_data` writes the per-epoch history to CSV, and
`save_plot` renders it with matplotlib.

Stage configs control the experiment axes. Defaults:

| Axis | Default | Ablation options |
| --- | --- | --- |
| `modalities` | `meta4` (EEG + HbO + HbR) | `EEG`, `HbO`, `HbR`, `meta1`, `meta2`, `meta3` |
| `dts_type` | `dts` | `no_dts` |
| `fusion_type` | `tgca` | `concat`, `co_attention`, `gated`, `lmf` |

`meta1 = HbO + HbR`, `meta2 = HbR + EEG`, `meta3 = HbO + EEG`,
`meta4 = HbO + HbR + EEG`. `dts = w/ DTS` and `no_dts = w/o DTS`.

### DTS

DTS applies a fixed 4 s advance (40 steps at 10 Hz after CNN pooling) plus a
learnable residual bounded to ±10 steps, i.e. an effective 3–5 s alignment
range. EEG is the anchor and is never shifted. With `no_dts`, the fNIRS branch
uses the first 2 s only, matching the temporal window of EEG.

### Stage I

5-fold GroupKFold at the trial level, 100 epochs, batch 32, Adam with
lr 1e-3 and L2 1e-4, 5-epoch warm-up, ReduceLROnPlateau, early stopping
patience 15.

### Stage II

LOSO with source-only validation: early stopping and the learning-rate
scheduler use a subject-wise split of the source subjects; target labels are
used only for final scoring. Pre-training is capped at 40 epochs with L2 1e-3
and early stopping patience 10.

Source selection is optional:

```yaml
source_selection:
  enabled: false   # true: run physio/data/performance/hybrid selection
```

When disabled, all source subjects are used and DA runs directly. The
`performance` and `hybrid` strategies require the Stage I summary CSV. The
`physio` strategy additionally requires `data/physio_metadata.csv` and is only
implemented for the public healthy cohort.

### Stage III

Few-shot personalization with `none`, `linear_probing`, `full`, `lora`, and
`dora`. Calibration and test sets are disjoint at the **trial** level; `shots`
is the number of calibration trials **per class**.

Batch 8, L2 1e-2, fixed 30 epochs (no early stopping or scheduler), lr 1e-3
for PEFT and 1e-4 for full fine-tuning. DoRA uses rank 2, alpha 8, and
dropout 0.1.

---

## 5. Results

```
results/
├── Stage1_public/<task>_<dts>_<modalities>_<fusion>/
│   ├── <exp>_Summary.csv
│   ├── <exp>_TestResults.xlsx
│   ├── checkpoints/sub_<id>/fold_<i>_best.pt
│   └── confusion_matrix/sub_<id>/
├── Stage2_public/<task>_<strategy>/
│   ├── LOSO_Summary.csv
│   └── target_<id>/best_model.pt
└── Stage3_public/<task>_<method>_<shots>shot/details.csv
```

Stage III reads Stage II checkpoints from the `pretrained` paths in its config.

---

## 6. Design notes

* Stage I and Stage II split by `trial_ids`; Stage III selects calibration
  trials and keeps their windows together.
