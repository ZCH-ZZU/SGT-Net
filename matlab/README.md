# MATLAB preprocessing for SGT-Net

This folder contains the MATLAB preprocessing pipeline for

**SGT-Net: A Three-Stage Framework for Cross-Subject Hybrid EEG–fNIRS Decoding
via Temporal Synchronization and Parameter-Efficient Personalization**

---

## 1. Protocol

The pipeline:

| Step | EEG | fNIRS |
| --- | --- | --- |
| Resampling | 200 Hz | 10 Hz |
| Filtering | 4–35 Hz, 6th-order zero-phase Butterworth | 0.01–0.1 Hz |
| Spatial processing | common average reference | — |
| Conversion | — | modified Beer–Lambert law (benchmark only) |
| Epoch | −5 to 20 s around stimulus onset | same |
| Baseline | −3 to 0 s | same |
| Windows | 2 s, starts 1–8 s, step 1 s | 8 s, starts 1–8 s, step 1 s |

---

## 2. Requirements

* MATLAB R2021b or newer
* Signal Processing Toolbox (`butter`, `filtfilt`, `zp2sos`, `resample`)
* HDF5 support, included in MATLAB (`h5create`, `h5write`, `h5writeatt`)

---

## 3. Folder structure

```
matlab/
├── run_preprocessing.m          % single entry point
├── verify_dataset.m             % HDF5 structure checker
├── README.md                    % this file
└── +sgtnet/                     % implementation package
    ├── default_config.m
    ├── build_public_dataset.m
    ├── build_stroke_dataset.m
    ├── process_public_subject.m
    ├── process_stroke_subject.m
    ├── load_public_raw.m
    ├── load_stroke_raw.m
    ├── preprocess_eeg.m
    ├── preprocess_nirs.m
    ├── regress_out_eog.m
    ├── subset_montage.m
    ├── beer_lambert.m
    ├── filter_bandpass.m
    ├── resample_3d.m
    ├── cut_epochs.m
    ├── baseline_correct.m
    ├── make_windows.m
    ├── zscore_windows.m
    ├── make_payload.m
    ├── h5_init.m
    └── h5_write_subject.m
```

`run_preprocessing.m` adds `matlab/` to the MATLAB path, so the package is
called as `sgtnet.function_name(...)`.

---

## 4. Raw data layout

By default, the pipeline expects the raw data under `../data/public` and
`../data/stroke`; see [`../data/README.md`](../data/README.md). The required
subfolders are shown below. If the data is stored elsewhere, pass
`PublicRoot` and `StrokeRoot` to `run_preprocessing`.

### 4.1 Benchmark dataset (Shin et al., 2017)

Download: <http://doc.ml.tu-berlin.de/hBCI>

`PublicRoot` must point to the folder that contains `EEG/` and `NIRS/`
(usually `<hybrid-BCI-master>/data`):

```
<PublicRoot>/
├── EEG/
│   └── subject 01/
│       └── with occular artifact/
│           ├── cnt.mat
│           ├── mrk.mat
│           └── mnt.mat
└── NIRS/
    └── subject 01/
        ├── cnt.mat
        ├── mrk.mat
        └── mnt.mat
```

There are 29 subjects with six sessions each (20 trials per session). MI uses
sessions 1, 3 and 5; MA uses sessions 2, 4 and 6.

### 4.2 Clinical dataset (Shi et al., 2025)

Download: <https://doi.org/10.6084/m9.figshare.28955456.v4>

`StrokeRoot` must point to the `epo_data` folder:

```
<StrokeRoot>/
├── Patients/
│   ├── EEG/subject1/1_epo.mat, 2_epo.mat, ...
│   └── NIRS/subject1/1_epo.mat, 2_epo.mat, ...
└── Normals/
    └── ...
```

Only `Patients/` is written. Subjects have 1–3 sessions (30 trials each), so a
subject contributes 30–90 trials.

---

## 5. Running the pipeline

```matlab
cd matlab

% With raw data in ../data/public and ../data/stroke
run_preprocessing('all', 'Overwrite', true)

% Or with explicit paths
run_preprocessing('all', ...
    'PublicRoot', 'D:/data/hybridBCI/data', ...
    'StrokeRoot', 'D:/data/HEFMI-ICH/epo_data', ...
    'OutputDir',  'D:/data/SGT-Net', ...
    'Overwrite',  true)

% One dataset
run_preprocessing('public', 'PublicRoot', 'D:/data/hybridBCI/data', ...
    'Overwrite', true)
run_preprocessing('stroke', 'StrokeRoot', 'D:/data/HEFMI-ICH/epo_data', ...
    'Overwrite', true)

% A few subjects, useful for a quick check
run_preprocessing('stroke', 'StrokeRoot', 'D:/data/HEFMI-ICH/epo_data', ...
    'Subjects', 1:3, 'Overwrite', true)
```

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `PublicRoot` | `<project>/data/public` | benchmark `data` folder |
| `StrokeRoot` | `<project>/data/stroke` | HEFMI-ICH `epo_data` folder |
| `OutputDir` | `<project>/datasets` | output folder |
| `Subjects` | all | subject IDs |
| `Overwrite` | `false` | delete an existing output file |
| `Verbose` | `true` | print progress |

Check a generated file with:

```matlab
verify_dataset('../datasets/hybridBCI_Tn8.h5')
verify_dataset('../datasets/stroke_patients_Tn8.h5')
```

---

## 6. Output files

### 6.1 `hybridBCI_Tn8.h5`

```
/
├── eeg_fs, nirs_fs, eeg_units, nirs_units
├── eeg_channels, nirs_channels
├── eeg_mnt_original/, nirs_mnt_original/
├── MI/
│   ├── class_names = ["Left", "Right"]
│   ├── eeg_channels, nirs_channels
│   ├── attributes: filter bands, orders, epoch, baseline, windows,
│   │               n_windows_per_trial = 8
│   └── subject_01/ ... subject_29/
│       ├── eeg_data        [400, 30, 480]  (time, channels, samples)
│       ├── hbo_data        [ 80, 36, 480]
│       ├── hbr_data        [ 80, 36, 480]
│       ├── labels_decimal  [  1, 480]      1-based
│       └── trial_ids       [  1, 480]      1-based
└── MA/
    ├── class_names = ["Rest", "Arithmetic"]
    └── subject_01/ ...
```

### 6.2 `stroke_patients_Tn8.h5`

```
/
├── eeg_fs, nirs_fs, eeg_units, nirs_units
├── eeg_channels (32), nirs_channels (90)
└── MI/
    ├── class_names = ["Left", "Right"]
    └── subject_01/ ... subject_20/
        ├── eeg_data        [400, 32, 720]  (up to 90 trials x 8 windows)
        ├── hbo_data        [ 80, 90, 720]
        ├── hbr_data        [ 80, 90, 720]
        ├── labels_decimal  [  1, 720]
        └── trial_ids       [  1, 720]
```

The number of samples varies per patient because the number of sessions varies
(1–3). Subject 8, for example, has one session and 240 samples.

---

## 7. Notes

* MATLAB's HDF5 layer reverses dimension order: a MATLAB array
  `[samples, channels, time]` is stored as `[time, channels, samples]`, which
  is the layout expected by the Python loaders. Labels and trial IDs are stored
  as `[1, samples]`; `pos_3d` is stored as `[channels, 3]`.
* `trial_ids` identifies the parent trial of every sliding window. Windows are
  stored in trial-major order.
* The benchmark release contains 2 EOG channels in addition to the 30 scalp
  EEG channels. EOG channels are always discarded from the model input. Set
  `cfg.eeg.eog_correction = true` in `default_config.m` to regress out ocular
  activity using the EOG channels before they are discarded (disabled by
  default; the manuscript does not include this step).
* The benchmark fNIRS data are raw optical intensities and are converted with
  the modified Beer–Lambert law. The clinical release already provides HbO/HbR
  concentration changes, which are converted to mmol/L to match the benchmark
  unit.

Expected totals:

| Dataset | Subjects | Trials | Windows/trial | Samples |
| --- | --- | --- | --- | --- |
| Benchmark MI | 29 | 1740 | 8 | 13,920 |
| Benchmark MA | 29 | 1740 | 8 | 13,920 |
| Clinical MI | 20 | 1440 | 8 | 11,520 |

---

## 8. Citations

If you use this code, please cite the SGT-Net manuscript and the two dataset
papers:

```bibtex
@article{shin2017,
  author  = {Shin, Jaeyoung and von L{\"u}hmann, Alexander and Blankertz, Benjamin
             and Kim, Do-Won and Jeong, Jichai and Hwang, Han-Jeong
             and M{\"u}ller, Klaus-Robert},
  title   = {Open Access Dataset for {EEG+NIRS} Single-Trial Classification},
  journal = {IEEE Trans. Neural Syst. Rehabil. Eng.},
  year    = {2017},
  volume  = {25},
  number  = {10},
  pages   = {1735--1745},
  doi     = {10.1109/TNSRE.2016.2628057}
}

@article{shi2025,
  author  = {Shi, Jian and Chen, Danyang and Zhao, Xingwei and Zhao, Zhixian
             and Li, Shengjie and Xu, Yeguang and Ding, Tao and Zhu, Zheng
             and Zhang, Peng and Ye, Qing and others},
  title   = {{HEFMI-ICH}: A Hybrid {EEG-fNIRS} Motor Imagery Dataset for
             Brain-Computer Interface in Intracerebral Hemorrhage},
  journal = {Scientific Data},
  year    = {2025},
  volume  = {12},
  number  = {1},
  pages   = {1816},
  doi     = {10.1038/s41597-025-06100-7}
}
```
